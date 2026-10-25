// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4 chat-template encoder — Swift port of
// `jang-tools/jang_tools/dsv4_prune/encoding/encoding_dsv4.py`
// (authoritative copy also shipped inside every DSV4 bundle at
// `encoding/encoding_dsv4.py`).
//
// DSV4 does NOT ship a `chat_template` in `tokenizer_config.json` —
// callers must use this encoder to build prompts. The encoder handles:
//
//   - BOS (`<｜begin▁of▁sentence｜>`) and EOS (`<｜end▁of▁sentence｜>`)
//   - User / Assistant / System / Developer / latest_reminder roles
//   - Two thinking modes: `chat` (empty closed `</think>` tail, model
//     generates direct answer) vs `thinking` (open `<think>` tail,
//     model generates reasoning then answer)
//   - Three reasoning-effort levels: nil / "high" / "max"
//   - `drop_earlier_reasoning`: multi-turn strips prior turns' think
//     blocks (unless any message still carries `tools`)
//   - Tool call encoding in DSML format (curly-quote U+FF5C markers)
//   - Tool result merging (`<tool_result>…</tool_result>`) into user
//     messages via `content_blocks`, which DSV4 has in place of the
//     standalone `tool` role
//
// Reference: `jang/research/DSV4-RUNTIME-ARCHITECTURE.md` §4
// ("Chat template + reasoning modes") and §11 (testing matrix).

import Foundation

// MARK: - Special tokens (curly-quote markers are U+FF5C, NOT ASCII `|`)

public enum DeepseekV4Tokens {
    public static let bos = "<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}>"
    public static let eos = "<\u{FF5C}end\u{2581}of\u{2581}sentence\u{FF5C}>"
    public static let user = "<\u{FF5C}User\u{FF5C}>"
    public static let assistant = "<\u{FF5C}Assistant\u{FF5C}>"
    /// DeepSeek-V4.1 only (`encoding/encoding.py` `SYSTEM_SP_TOKEN`). The 0731
    /// contract never renders it.
    public static let system = "<\u{FF5C}System\u{FF5C}>"
    public static let latestReminder = "<\u{FF5C}latest_reminder\u{FF5C}>"
    public static let thinkStart = "<think>"
    public static let thinkEnd = "</think>"
    /// Curly-quote prefix/suffix chunk used inside DSML blocks. Note
    /// the tag pattern is `<｜DSML｜something>` — the markers are
    /// U+FF5C (fullwidth vertical bar), not ASCII `|`.
    public static let dsml = "\u{FF5C}DSML\u{FF5C}"

    /// Task special tokens for internal classification workflows.
    /// Kept here for completeness; the primary chat path doesn't use them.
    public static let taskSPTokens: [String: String] = [
        "action": "<\u{FF5C}action\u{FF5C}>",
        "query": "<\u{FF5C}query\u{FF5C}>",
        "authority": "<\u{FF5C}authority\u{FF5C}>",
        "domain": "<\u{FF5C}domain\u{FF5C}>",
        "title": "<\u{FF5C}title\u{FF5C}>",
        "read_url": "<\u{FF5C}read_url\u{FF5C}>",
    ]
}

// MARK: - Public enums

public enum DeepseekV4ThinkingMode: String, Sendable {
    case chat
    case thinking
}

public enum DeepseekV4ReasoningEffort: String, Sendable {
    case low
    case high
    case max
}

/// Which DeepSeek-V4 prompt contract a bundle was trained on.
///
/// DSV4-0731 (`model_type: deepseek_v4`) and DeepSeek-V4.1-1017
/// (`model_type: deepseek_v41`, text decoder `deepseek_v41_text`) share the
/// BOS/EOS tokens the tokenizer bridges sniff, but their wire formats differ:
/// V4.1 renders a numeric reasoning-effort budget inside `<｜System｜>`,
/// leads system messages with `<｜System｜>`, supports mid-conversation system
/// turns, and names its DSML tags `" calls"` / `" invoke"` / `" parameter"`
/// (leading space). The family therefore has to come from the bundle's
/// config, never from the tokenizer. `.deepseekV4` is the default everywhere
/// so a caller that does not thread a family keeps the 0731 bytes exactly.
public enum DeepseekV4ChatFamily: String, Sendable, Hashable {
    case deepseekV4 = "deepseek_v4"
    case deepseekV41 = "deepseek_v41"

    /// Map a config `model_type` (top-level or `text_config.model_type`).
    /// Anything that is not explicitly V4.1 stays on the 0731 contract.
    public init(modelType: String?) {
        let normalized = (modelType ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
        if normalized == "deepseek_v41" || normalized == "deepseek_v41_text"
            || normalized.hasPrefix("deepseek_v41_")
        {
            self = .deepseekV41
        } else {
            self = .deepseekV4
        }
    }

    /// Resolve from raw `config.json` bytes: top-level `model_type`, then
    /// `text_config.model_type`.
    public static func detect(configData: Data) -> DeepseekV4ChatFamily {
        guard let object = try? JSONSerialization.jsonObject(with: configData)
            as? [String: Any]
        else { return .deepseekV4 }
        if DeepseekV4ChatFamily(modelType: object["model_type"] as? String) == .deepseekV41 {
            return .deepseekV41
        }
        let text = object["text_config"] as? [String: Any]
        return DeepseekV4ChatFamily(modelType: text?["model_type"] as? String)
    }

    /// Resolve from a bundle directory's `config.json`.
    public static func detect(modelDirectory: URL) -> DeepseekV4ChatFamily {
        guard let data = try? Data(
            contentsOf: modelDirectory.appendingPathComponent("config.json"))
        else { return .deepseekV4 }
        return detect(configData: data)
    }

    /// DSML outer block tag name (`<｜DSML｜{name}>`).
    public var dsmlCallsBlockName: String {
        self == .deepseekV41 ? " calls" : "tool_calls"
    }

    /// DSML per-call tag name.
    public var dsmlInvokeTagName: String {
        self == .deepseekV41 ? " invoke" : "invoke"
    }

    /// DSML per-parameter tag name.
    public var dsmlParameterTagName: String {
        self == .deepseekV41 ? " parameter" : "parameter"
    }
}

/// A tokenizer bridge that routes DSV4-family prompts through
/// ``DeepseekV4ChatEncoder`` and can be told which prompt contract the bundle
/// uses. Tokenizer loaders that know the model directory call
/// ``settingDeepseekV4ChatFamily(_:)`` after construction; bridges that are
/// never told keep the 0731 contract.
public protocol DeepseekV4ChatFamilyConfigurableTokenizer: Tokenizer {
    func settingDeepseekV4ChatFamily(_ family: DeepseekV4ChatFamily) -> any Tokenizer
}

// MARK: - Encoder

public struct DeepseekV4ChatEncoder: Sendable {

    /// Prompt contract to render. Defaults to DSV4-0731 so existing callers
    /// are byte-for-byte unchanged; V4.1 must be requested explicitly.
    public let family: DeepseekV4ChatFamily

    public init(family: DeepseekV4ChatFamily = .deepseekV4) {
        self.family = family
    }

    // MARK: DeepSeek-V4.1 reasoning-effort contract

    // V4.1 (1017) renders a NUMERIC budget inside the System token, per the
    // bundle's `encoding/encoding.py` (`REASONING_EFFORT_TEMPLATE` +
    // `render_reasoning_effort`), at index 0 in thinking mode only. The 0731
    // text prefaces below are the wrong contract for V4.1 (the model was never
    // trained on them and meta-reasons about "complying with a thinking
    // block", degenerating into repetition loops on prompts such as "Say hello
    // in exactly one word."). They stay the 0731 contract.
    static let deepseekV41ReasoningEffortTemplate =
        "Reasoning Effort: %d (range 1-100, the higher the value, the more thorough the reasoning)\n\n"

    /// low/high/max → numeric budget (encoding.py `REASONING_EFFORT_MAPPINGS`).
    static func deepseekV41ReasoningEffortBudget(_ effort: DeepseekV4ReasoningEffort) -> Int {
        switch effort {
        case .low: return 50
        case .high: return 75
        case .max: return 100
        }
    }

    /// Budget used when a V4.1 request carries no effort at all.
    ///
    /// The bundle reference defaults to `"high"` (75, encoding.py
    /// `DEFAULT_REASONING_EFFORT`). vmlx deliberately keeps the host UX of
    /// "absent effort ≡ low" (50) so the default reasoning chip ("Low") and
    /// the rendered contract agree, matching the 0731 nil ≡ .low rule. Pass an
    /// explicit effort (or numeric budget) to get the reference default.
    public static let deepseekV41DefaultReasoningBudget = 50

    // `high`/`max` are verbatim 0731 `REASONING_EFFORT_PROMPTS` values —
    // part of the model's prompt contract; never reword or interchange them.
    //
    // The official contract defines `low` as the EMPTY STRING: no directive
    // is injected and the model self-modulates think depth, so hard first
    // prompts still produce long reasoning on the "Low" rail. That is
    // unacceptable for surfaces whose default promises low reasoning, so
    // vmlx deliberately deviates: `.low` (and absent effort, which the
    // official DEFAULT_REASONING_EFFORT maps to low) injects a style-matched
    // enforcement preface. nil ≡ .low by construction so warmup and send
    // renders can never diverge.
    static let reasoningEffortLowPreface = """
        Reasoning Effort: Low — minimal thinking only.
        Begin with a very short thinking block: one or two sentences confirming your approach, never more.
        Then close your thinking and give only the final answer — clean and direct, without restating your reasoning.


        """

    static let reasoningEffortHighPreface = """
        Reasoning Effort: Absolute maximum with no shortcuts permitted.
        You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.
        Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.


        """

    static let reasoningEffortMaxPreface = """
        Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.
        You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.
        Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.


        """

    // MARK: - Public API

    /// Encode a list of `Message` structs into the DSV4 prompt format.
    /// - Parameters:
    ///   - messages: conversation history, OpenAI-style roles + fields.
    ///   - thinkingMode: `.chat` or `.thinking` (DSV4 has these two).
    ///   - reasoningEffort: nil ≡ `.low` / `.high` / `.max`. Only applies
    ///     in `.thinking` mode; every rail injects its preface at index 0
    ///     (low is a deliberate vmlx enforcement extension — official
    ///     contract leaves it empty).
    ///   - dropEarlierReasoning: strip `reasoning_content` from prior
    ///     assistant turns. Forced `false` whenever any message still
    ///     carries `tools` (mid-trajectory agent turns need full CoT).
    ///   - context: optional pre-rendered prefix (for cached contexts);
    ///     not prepended to output but used for index accounting.
    ///   - addBOS: prepend `<｜begin▁of▁sentence｜>` when no context.
    public func encode(
        messages: [Message],
        thinkingMode: DeepseekV4ThinkingMode = .thinking,
        reasoningEffort: DeepseekV4ReasoningEffort? = nil,
        dropEarlierReasoning: Bool = true,
        context: [Message] = [],
        addBOS: Bool = true,
        toolChoiceRequired: Bool = false,
        toolChoiceName: String? = nil,
        reasoningBudget: Int? = nil
    ) -> String {
        // `reasoningBudget` is the DeepSeek-V4.1 numeric effort (1-100). It
        // wins over `reasoningEffort` on the V4.1 contract and is ignored on
        // the 0731 contract, which has no numeric budget.
        // Preprocess: merge tool messages into user, sort tool_result
        // blocks by the order they were called in the prior assistant.
        var processedMessages = Self.mergeToolMessages(messages)
        var processedContext = Self.mergeToolMessages(context)
        let merged = Self.sortToolResultsByCallOrder(processedContext + processedMessages)
        processedMessages = Array(merged[processedContext.count...])
        processedContext = Array(merged[..<processedContext.count])

        let full = processedContext + processedMessages

        var prompt = (addBOS && context.isEmpty) ? DeepseekV4Tokens.bos : ""

        // Resolve drop_thinking: if any message still carries tools,
        // do NOT strip (mid-agent trajectory needs reasoning).
        var effectiveDrop = dropEarlierReasoning
        if full.contains(where: { !($0.tools?.isEmpty ?? true) }) {
            effectiveDrop = false
        }

        let rendered: [Message]
        let contextLen: Int
        if thinkingMode == .thinking && effectiveDrop {
            rendered = Self.dropThinkingMessages(full, family: family)
            contextLen = rendered.count - processedMessages.count
        } else {
            rendered = full
            contextLen = processedContext.count
        }

        var finalRendered = rendered
        if toolChoiceRequired,
            finalRendered.contains(where: { !($0.tools?.isEmpty ?? true) }),
            finalRendered.last?.role != .latestReminder
        {
            let requiredToolName =
                Self.normalizedToolChoiceName(toolChoiceName)
                ?? Self.singleAvailableToolName(in: finalRendered)
            let reminder = Self.requiredToolChoiceReminder(
                toolName: requiredToolName, family: family)
            if let tailIndex = finalRendered.lastIndex(where: {
                $0.role == .user || $0.role == .developer
            }), tailIndex >= contextLen {
                finalRendered.insert(reminder, at: finalRendered.index(after: tailIndex))
            } else {
                finalRendered.append(reminder)
            }
        }

        for i in 0..<(finalRendered.count - contextLen) {
            prompt += renderMessage(
                at: i + contextLen,
                in: finalRendered,
                thinkingMode: thinkingMode,
                dropThinking: effectiveDrop,
                reasoningEffort: reasoningEffort,
                reasoningBudget: reasoningBudget,
                toolChoiceRequired: toolChoiceRequired,
                toolChoiceName: toolChoiceName
            )
        }

        return prompt
    }

    /// Render OpenAI-style chat dictionaries through DSV4's native encoder.
    ///
    /// DSV4 bundles intentionally ship `encoding/encoding_dsv4.py` instead of
    /// a tokenizer Jinja template.  Tokenizer bridges receive the generic
    /// OpenAI dictionary surface, so this adapter mirrors the Python runtime's
    /// boundary before calling the strongly typed encoder above:
    ///
    /// - tool schemas are attached to the leading system message (or an empty
    ///   system message is inserted), exactly like the Python adapter;
    /// - tool-call argument dictionaries are serialized to the JSON string
    ///   consumed by the official encoder;
    /// - `enable_thinking` and the 0731 low/high/max effort values select the
    ///   native chat/thinking rails;
    /// - `addGenerationPrompt=false` removes only the terminal assistant rail.
    ///
    /// Keeping this conversion in MLXLMCommon avoids a second, subtly different
    /// DSV4 prompt implementation in the Hugging Face tokenizer bridge.
    ///
    /// `family` selects the prompt contract (resolve it from the bundle with
    /// ``DeepseekV4ChatFamily/detect(modelDirectory:)``). The default is the
    /// 0731 contract. On `.deepseekV41` the effort accepts the full host ladder
    /// (`none/minimal/low/medium/high/xhigh/max`) or an explicit 1-100 number;
    /// see ``reasoningBudget(from:)``.
    public static func renderOpenAIChat(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?,
        addGenerationPrompt: Bool,
        family: DeepseekV4ChatFamily = .deepseekV4
    ) throws -> String {
        var nativeMessages = try messages.map(Self.messageFromOpenAI)

        if let tools, !tools.isEmpty {
            let systemWithTools = nativeMessages.contains {
                $0.role == .system && !($0.tools?.isEmpty ?? true)
            }
            if !systemWithTools {
                if nativeMessages.first?.role == .system {
                    nativeMessages[0].tools = tools
                } else {
                    nativeMessages.insert(
                        Message(role: .system, content: "", tools: tools),
                        at: 0
                    )
                }
            }
        }

        let effort = Self.reasoningEffort(from: additionalContext)
        let explicitlyEnabled = additionalContext?["enable_thinking"] as? Bool
        // DSV4's bundle contract (jang_config chat.reasoning) declares
        // default_mode="thinking" with default_effort="low". Absent controls
        // therefore mean the thinking rail on the low preface (nil ≡ .low
        // in the encoder) — the same default `encode(thinkingMode:)` uses.
        // Only an explicit enable_thinking=false selects the chat rail;
        // defaulting the absent case to chat rendered a closed </think>
        // tail while the UI reported the "Low" thinking default.
        let thinkingMode: DeepseekV4ThinkingMode =
            explicitlyEnabled == false ? .chat : .thinking
        let dropEarlierReasoning =
            (additionalContext?["drop_earlier_reasoning"] as? Bool) ?? true
        let (toolChoiceRequired, toolChoiceName) = Self.toolChoice(
            from: additionalContext?["tool_choice"])

        var prompt = DeepseekV4ChatEncoder(family: family).encode(
            messages: nativeMessages,
            thinkingMode: thinkingMode,
            reasoningEffort: thinkingMode == .thinking ? effort : nil,
            dropEarlierReasoning: dropEarlierReasoning,
            toolChoiceRequired: toolChoiceRequired,
            toolChoiceName: toolChoiceName,
            reasoningBudget: (family == .deepseekV41 && thinkingMode == .thinking)
                ? Self.reasoningBudget(from: additionalContext) : nil
        )
        if !addGenerationPrompt {
            prompt = Self.removingTerminalGenerationRail(from: prompt)
        }
        return prompt
    }

    private static func messageFromOpenAI(
        _ raw: [String: any Sendable]
    ) throws -> Message {
        guard let roleText = raw["role"] as? String,
            let role = MessageRole(rawValue: roleText)
        else {
            throw OpenAIAdapterError.unsupportedRole(
                raw["role"].map(String.init(describing:)) ?? "nil")
        }

        let toolCalls: [ToolCall]?
        if let rawCalls = raw["tool_calls"] as? [[String: any Sendable]] {
            let rawArguments = raw[rawToolArgumentsJSONMessageKey] as? [String]
            toolCalls = try rawCalls.enumerated().map { index, call in
                let preserved = rawArguments.flatMap { values in
                    index < values.count ? values[index] : nil
                }
                return try Self.toolCallFromOpenAI(
                    call,
                    rawArgumentsJSON: preserved
                )
            }
        } else {
            toolCalls = nil
        }

        return Message(
            role: role,
            content: Self.textContent(raw["content"]),
            contentBlocks: Self.contentBlocks(raw["content_blocks"]),
            reasoningContent: raw["reasoning_content"] as? String,
            toolCalls: toolCalls,
            toolCallId: raw["tool_call_id"] as? String,
            tools: raw["tools"] as? [[String: any Sendable]],
            responseFormat: raw["response_format"] as? [String: any Sendable],
            task: raw["task"] as? String,
            woEOS: (raw["wo_eos"] as? Bool) ?? false
        )
    }

    private static func toolCallFromOpenAI(
        _ raw: [String: any Sendable],
        rawArgumentsJSON: String? = nil
    ) throws -> ToolCall {
        let function = (raw["function"] as? [String: any Sendable]) ?? raw
        guard let name = function["name"] as? String, !name.isEmpty else {
            throw OpenAIAdapterError.missingToolName
        }
        let arguments: String
        if let preserved = validatedRawArgumentsJSON(
            rawArgumentsJSON,
            matching: function["arguments"]
        ) {
            arguments = preserved
        } else if let string = function["arguments"] as? String {
            arguments = string
        } else if let dictionary = function["arguments"] as? [String: any Sendable] {
            arguments = dictionary.jsonSerialized()
        } else if let value = function["arguments"],
            let data = try? JSONSerialization.data(
                withJSONObject: value, options: [.withoutEscapingSlashes])
        {
            arguments = String(data: data, encoding: .utf8) ?? "{}"
        } else {
            arguments = "{}"
        }
        return ToolCall(id: raw["id"] as? String, name: name, arguments: arguments)
    }

    private static func validatedRawArgumentsJSON(
        _ candidate: String?,
        matching value: (any Sendable)?
    ) -> String? {
        guard let candidate, !candidate.isEmpty,
            let data = candidate.data(using: .utf8),
            let decoded = try? JSONDecoder().decode(
                [String: JSONValue].self,
                from: data
            )
        else { return nil }

        if let dictionary = value as? [String: any Sendable] {
            guard decoded == dictionary.mapValues({ JSONValue.from($0) }) else {
                return nil
            }
        } else if let string = value as? String,
            let expectedData = string.data(using: .utf8),
            let expected = try? JSONDecoder().decode(
                [String: JSONValue].self,
                from: expectedData
            )
        {
            guard decoded == expected else { return nil }
        } else {
            return nil
        }
        return candidate
    }

    private static func textContent(_ value: (any Sendable)?) -> String? {
        if let string = value as? String { return string }
        guard let parts = value as? [[String: any Sendable]] else { return nil }
        let text = parts.compactMap { part -> String? in
            guard let type = part["type"] as? String,
                ["text", "input_text", "output_text"].contains(type)
            else { return nil }
            return (part["text"] as? String) ?? (part["content"] as? String)
        }.joined(separator: "\n\n")
        return text.isEmpty ? nil : text
    }

    private static func contentBlocks(
        _ value: (any Sendable)?
    ) -> [MessageContentBlock]? {
        guard let blocks = value as? [[String: any Sendable]] else { return nil }
        let converted = blocks.compactMap { block -> MessageContentBlock? in
            switch block["type"] as? String {
            case "text":
                return .text((block["text"] as? String) ?? "")
            case "tool_result":
                return .toolResult(
                    toolUseId: (block["tool_use_id"] as? String) ?? "",
                    content: Self.textContent(block["content"]) ?? ""
                )
            default:
                return nil
            }
        }
        return converted.isEmpty ? nil : converted
    }

    private static func reasoningEffort(
        from context: [String: any Sendable]?
    ) -> DeepseekV4ReasoningEffort? {
        guard let raw = context?["reasoning_effort"] as? String else { return nil }
        switch raw.lowercased() {
        case "low": return .low
        case "high": return .high
        case "max": return .max
        default: return nil
        }
    }

    /// Resolve the DeepSeek-V4.1 numeric reasoning-effort budget (1-100) from
    /// the request so every host slider position takes effect, not just the
    /// three named anchors the enum carries. Contract anchors are low=50 /
    /// high=75 / max=100 (encoding.py `REASONING_EFFORT_MAPPINGS`); the
    /// in-between rungs of the host ladder interpolate. An explicit Int /
    /// Double / numeric string is clamped to 1...100. Returns nil when no
    /// effort was requested (the encoder then uses
    /// ``deepseekV41DefaultReasoningBudget``). V4.1 only; the 0731 contract
    /// never consults this.
    public static func reasoningBudget(from context: [String: any Sendable]?) -> Int? {
        guard let rawAny = context?["reasoning_effort"] else { return nil }
        if let i = rawAny as? Int { return max(1, min(100, i)) }
        if let d = rawAny as? Double, d.isFinite { return max(1, min(100, Int(d.rounded()))) }
        guard let raw = (rawAny as? String)?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        else { return nil }
        if let i = Int(raw) { return max(1, min(100, i)) }
        switch raw {
        case "none", "minimal": return 25
        case "low": return 50
        case "medium": return 62
        case "high": return 75
        case "xhigh": return 88
        case "max": return 100
        default: return nil
        }
    }

    private static func toolChoice(
        from raw: (any Sendable)?
    ) -> (required: Bool, name: String?) {
        if let string = raw as? String {
            return (string == "required", nil)
        }
        guard let object = raw as? [String: any Sendable] else {
            return (false, nil)
        }
        let function = object["function"] as? [String: any Sendable]
        if (object["type"] as? String) == "function",
            let name = function?["name"] as? String
        {
            return (true, name)
        }
        return (false, nil)
    }

    private static func removingTerminalGenerationRail(from prompt: String) -> String {
        let rails = [
            DeepseekV4Tokens.assistant + DeepseekV4Tokens.thinkStart,
            DeepseekV4Tokens.assistant + DeepseekV4Tokens.thinkEnd,
        ]
        for rail in rails where prompt.hasSuffix(rail) {
            return String(prompt.dropLast(rail.count))
        }
        return prompt
    }

    public enum OpenAIAdapterError: Error, CustomStringConvertible {
        case unsupportedRole(String)
        case missingToolName

        public var description: String {
            switch self {
            case .unsupportedRole(let role):
                return "Unsupported DSV4 chat role: \(role)"
            case .missingToolName:
                return "DSV4 tool call is missing function.name"
            }
        }
    }

    // MARK: - Message rendering

    func renderMessage(
        at index: Int,
        in messages: [Message],
        thinkingMode: DeepseekV4ThinkingMode,
        dropThinking: Bool,
        reasoningEffort: DeepseekV4ReasoningEffort?,
        reasoningBudget: Int? = nil,
        toolChoiceRequired: Bool,
        toolChoiceName: String?
    ) -> String {
        let msg = messages[index]
        let lastUserIdx = Self.findLastUserIndex(messages, family: family)
        let isV41 = family == .deepseekV41
        var out = ""

        if isV41 {
            // DeepSeek-V4.1 (encoding.py `render_message`): the numeric budget
            // at index 0 in thinking mode, and the System token LEADS the
            // conversation when there is an effort prefix OR the first message
            // is a system message. An explicit numeric budget (host ladder)
            // wins over the named enum; nil ≡ the documented vmlx default.
            var effortPrefix = ""
            if index == 0 && thinkingMode == .thinking {
                let budget = max(1, min(100,
                    reasoningBudget
                        ?? reasoningEffort.map(Self.deepseekV41ReasoningEffortBudget)
                        ?? Self.deepseekV41DefaultReasoningBudget))
                effortPrefix = String(format: Self.deepseekV41ReasoningEffortTemplate, budget)
            }
            if index == 0 && (!effortPrefix.isEmpty || msg.role == .system) {
                out += DeepseekV4Tokens.system
            }
            out += effortPrefix
        } else if index == 0 && thinkingMode == .thinking {
            // DSV4-0731: reasoning-effort preface only at index 0 in thinking
            // mode. Every rail injects its preface — including low (nil ≡
            // .low), which the official contract leaves empty; see
            // reasoningEffortLowPreface.
            switch reasoningEffort ?? .low {
            case .low:
                out += Self.reasoningEffortLowPreface
            case .high:
                out += Self.reasoningEffortHighPreface
            case .max:
                out += Self.reasoningEffortMaxPreface
            }
        }

        switch msg.role {
        case .system:
            // V4.1 mid-conversation system message: its own System token lead
            // (at index 0 the lead was emitted above). 0731 has no System token.
            if isV41 && index > 0 {
                out += DeepseekV4Tokens.system
            }
            out += msg.content ?? ""
            if let tools = msg.tools, !tools.isEmpty {
                out += "\n\n" + Self.renderTools(
                    tools,
                    toolChoiceRequired: toolChoiceRequired,
                    toolChoiceName: toolChoiceName,
                    family: family
                )
            }
            if let rf = msg.responseFormat {
                out += "\n\n" + Self.renderResponseFormat(rf)
            }

        case .developer:
            var s = DeepseekV4Tokens.user
            s += msg.content ?? ""
            if let tools = msg.tools, !tools.isEmpty {
                s += "\n\n" + Self.renderTools(
                    tools,
                    toolChoiceRequired: toolChoiceRequired,
                    toolChoiceName: toolChoiceName,
                    family: family
                )
            }
            if let rf = msg.responseFormat {
                s += "\n\n" + Self.renderResponseFormat(rf)
            }
            out += s

        case .user:
            out += DeepseekV4Tokens.user
            if let blocks = msg.contentBlocks, !blocks.isEmpty {
                var parts: [String] = []
                for b in blocks {
                    switch b {
                    case .text(let s): parts.append(s)
                    case .toolResult(_, let content):
                        parts.append("<tool_result>\(content)</tool_result>")
                    }
                }
                out += parts.joined(separator: "\n\n")
            } else {
                out += msg.content ?? ""
            }

        case .latestReminder:
            out += DeepseekV4Tokens.latestReminder + (msg.content ?? "")

        case .tool:
            // DSV4 merges tool messages into user; caller should
            // preprocess. We never reach here after mergeToolMessages.
            break

        case .assistant:
            var thinkingPart = ""
            var toolCallContent = ""

            // Render tool calls (if any) in DSML format.
            if let tcs = msg.toolCalls, !tcs.isEmpty {
                let blocks = tcs.map { tc in
                    Self.renderToolCallInvoke(
                        name: tc.name, arguments: tc.arguments, family: family)
                }
                let joined = blocks.joined(separator: "\n")
                let block = family.dsmlCallsBlockName
                toolCallContent =
                    "\n\n<\(DeepseekV4Tokens.dsml)\(block)>\n\(joined)\n</\(DeepseekV4Tokens.dsml)\(block)>"
            }

            let content = msg.content ?? ""
            let reasoning = msg.reasoningContent ?? ""

            // A previous user message with a `task` means this is a
            // task-output assistant turn — no thinking block even in
            // thinking mode.
            let prevHasTask = index - 1 >= 0 && messages[index - 1].task != nil

            if thinkingMode == .thinking && !prevHasTask {
                if !dropThinking || index > lastUserIdx {
                    thinkingPart = reasoning + DeepseekV4Tokens.thinkEnd
                }
            }

            var piece = thinkingPart + content + toolCallContent
            if !msg.woEOS {
                piece += DeepseekV4Tokens.eos
            }
            out += piece
        }

        // Append transition tokens (Assistant marker + opening/closing
        // `<think>`/`</think>`) when the current message is the tail
        // of a user/developer turn OR there's a following assistant.
        let nextRole: MessageRole? =
            (index + 1 < messages.count) ? messages[index + 1].role : nil
        if nextRole != nil && nextRole != .assistant && nextRole != .latestReminder {
            return out
        }

        if let task = msg.task {
            let taskSP = DeepseekV4Tokens.taskSPTokens[task] ?? ""
            if task != "action" {
                out += taskSP
            } else {
                out += DeepseekV4Tokens.assistant
                out += thinkingMode == .thinking
                    ? DeepseekV4Tokens.thinkStart
                    : DeepseekV4Tokens.thinkEnd
                out += taskSP
            }
        } else if msg.role == .user || msg.role == .developer || msg.role == .latestReminder
            // V4.1: a mid-conversation system message behaves like a user
            // turn for the assistant generation header (encoding.py:731).
            || (isV41 && msg.role == .system && index > 0)
        {
            out += DeepseekV4Tokens.assistant
            // Decision tree for the final tail tag:
            //   dropThinking=false + thinking → `<think>` (open)
            //   dropThinking=true  + thinking + index>=lastUser → `<think>` (open)
            //   otherwise → `</think>` (closed, chat mode)
            if !dropThinking && thinkingMode == .thinking {
                out += DeepseekV4Tokens.thinkStart
            } else if dropThinking && thinkingMode == .thinking && index >= lastUserIdx {
                out += DeepseekV4Tokens.thinkStart
            } else {
                out += DeepseekV4Tokens.thinkEnd
            }
        }

        return out
    }

    // MARK: - Tools + schema rendering

    static let toolsTemplate = """
        ## Tools

        You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<\(DeepseekV4Tokens.dsml)tool_calls>" block like the following:

        <\(DeepseekV4Tokens.dsml)tool_calls>
        <\(DeepseekV4Tokens.dsml)invoke name="$TOOL_NAME">
        <\(DeepseekV4Tokens.dsml)parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</\(DeepseekV4Tokens.dsml)parameter>
        ...
        </\(DeepseekV4Tokens.dsml)invoke>
        <\(DeepseekV4Tokens.dsml)invoke name="$TOOL_NAME2">
        ...
        </\(DeepseekV4Tokens.dsml)invoke>
        </\(DeepseekV4Tokens.dsml)tool_calls>

        String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

        If thinking_mode is enabled (triggered by \(DeepseekV4Tokens.thinkStart)), you MUST output your complete reasoning inside \(DeepseekV4Tokens.thinkStart)...\(DeepseekV4Tokens.thinkEnd) BEFORE any tool calls or final response.

        Otherwise, output directly after \(DeepseekV4Tokens.thinkEnd) with tool calls or final response.

        ### Available Tool Schemas

        %@

        You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
        """

    /// DeepSeek-V4.1 `TOOLS_TEMPLATE` (encoding.py:481): identical prose, but
    /// the DSML tag names carry a leading space (`<｜DSML｜ calls>`,
    /// `<｜DSML｜ invoke …>`, `<｜DSML｜ parameter …>`).
    static let deepseekV41ToolsTemplate: String = {
        let f = DeepseekV4ChatFamily.deepseekV41
        let d = DeepseekV4Tokens.dsml
        let calls = f.dsmlCallsBlockName
        let invoke = f.dsmlInvokeTagName
        let parameter = f.dsmlParameterTagName
        return """
            ## Tools

            You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<\(d)\(calls)>" block like the following:

            <\(d)\(calls)>
            <\(d)\(invoke) name="$TOOL_NAME">
            <\(d)\(parameter) name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</\(d)\(parameter)>
            ...
            </\(d)\(invoke)>
            <\(d)\(invoke) name="$TOOL_NAME2">
            ...
            </\(d)\(invoke)>
            </\(d)\(calls)>

            String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

            If thinking_mode is enabled (triggered by \(DeepseekV4Tokens.thinkStart)), you MUST output your complete reasoning inside \(DeepseekV4Tokens.thinkStart)...\(DeepseekV4Tokens.thinkEnd) BEFORE any tool calls or final response.

            Otherwise, output directly after \(DeepseekV4Tokens.thinkEnd) with tool calls or final response.

            ### Available Tool Schemas

            %@

            You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
            """
    }()

    static func renderTools(
        _ tools: [[String: any Sendable]],
        toolChoiceRequired: Bool = false,
        toolChoiceName: String? = nil,
        family: DeepseekV4ChatFamily = .deepseekV4
    ) -> String {
        let schemas = tools.map { Self.functionSpec(from: $0) }
        let schemaJson = schemas.map { $0.jsonSerialized() }
        let schemasBlock = schemaJson.joined(separator: "\n")
        // Python's triple-quoted TOOLS_TEMPLATE includes a terminal newline.
        // Preserve it: it is the native separator before the next User token.
        let template = family == .deepseekV41 ? deepseekV41ToolsTemplate : toolsTemplate
        var rendered = template.replacingOccurrences(of: "%@", with: schemasBlock) + "\n"
        if toolChoiceRequired {
            let requiredToolName =
                Self.normalizedToolChoiceName(toolChoiceName)
                ?? Self.singleAvailableToolName(in: tools)
            let namedInstruction = requiredToolName.map {
                " Use the `\($0)` function."
            } ?? ""
            rendered +=
                "\n\nThe current assistant response MUST be a tool call. "
                + "Start with a \"<\(DeepseekV4Tokens.dsml)\(family.dsmlCallsBlockName)>\" block "
                + "and do not answer in prose before the tool result."
                + namedInstruction
        }
        return rendered
    }

    static func requiredToolChoiceReminder(
        toolName: String? = nil,
        family: DeepseekV4ChatFamily = .deepseekV4
    ) -> Message {
        let namedInstruction = toolName.map {
            " Use the `\($0)` function."
        } ?? ""
        return Message(
            role: .latestReminder,
            content:
                "The active API tool_choice is required for this assistant turn. "
                + "Call exactly one available tool now using a <\(DeepseekV4Tokens.dsml)\(family.dsmlCallsBlockName)> block. "
                + "Do not answer in prose before the tool result."
                + namedInstruction
        )
    }

    static func normalizedToolChoiceName(_ name: String?) -> String? {
        guard let name else { return nil }
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    static func singleAvailableToolName(in messages: [Message]) -> String? {
        var names: Set<String> = []
        for message in messages {
            guard let tools = message.tools else { continue }
            for tool in tools {
                if let name = Self.functionSpec(from: tool)["name"] as? String,
                    let normalized = Self.normalizedToolChoiceName(name)
                {
                    names.insert(normalized)
                }
            }
        }
        return names.count == 1 ? names.first : nil
    }

    static func singleAvailableToolName(in tools: [[String: any Sendable]]) -> String? {
        var names: Set<String> = []
        for tool in tools {
            if let name = Self.functionSpec(from: tool)["name"] as? String,
                let normalized = Self.normalizedToolChoiceName(name)
            {
                names.insert(normalized)
            }
        }
        return names.count == 1 ? names.first : nil
    }

    static func renderResponseFormat(_ rf: [String: any Sendable]) -> String {
        let json = rf.jsonSerialized()
        return
            "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n\(json)"
    }

    /// Extract the `function` sub-dict from OpenAI-format tool specs.
    /// Matches `tools_from_openai_format` in encoding_dsv4.py — when
    /// a tool is wrapped in `{"type":"function","function":{…}}` we
    /// peel the wrapper; otherwise pass through.
    static func functionSpec(from tool: [String: any Sendable]) -> [String: any Sendable] {
        if let fn = tool["function"] as? [String: any Sendable] {
            return fn
        }
        return tool
    }

    // MARK: - Tool-call invoke rendering (DSML encode)

    /// Render a single `<｜DSML｜invoke>` block for one tool call.
    /// `arguments` is a JSON string (OpenAI convention) or already-decoded
    /// dict. Python reference accepts either.
    public static func renderToolCallInvoke(
        name: String,
        arguments: String,
        family: DeepseekV4ChatFamily = .deepseekV4
    ) -> String {
        var arguments = arguments
        if family == .deepseekV41,
            let data = arguments.data(using: .utf8),
            let inner = try? JSONSerialization.jsonObject(
                with: data, options: [.fragmentsAllowed]) as? String,
            let innerData = inner.data(using: .utf8),
            (try? JSONSerialization.jsonObject(with: innerData)) is [String: Any]
        {
            // V4.1 `encode_arguments_to_dsml` tolerates double-encoded JSON
            // argument strings (encoding.py:533-544). 0731 does not.
            arguments = inner
        }
        if let ordered = orderedJSONParameters(arguments) {
            return renderToolCallInvoke(name: name, orderedParameters: ordered, family: family)
        }
        let params: [String: Any]
        if let data = arguments.data(using: .utf8),
            let parsed = try? JSONSerialization.jsonObject(with: data, options: [])
                as? [String: Any]
        {
            params = parsed
        } else {
            // Malformed JSON — wrap raw into an "arguments" field so
            // the DSML envelope still round-trips.
            params = ["arguments": arguments]
        }
        return renderToolCallInvoke(name: name, params: params, family: family)
    }

    private struct OrderedJSONParameter {
        let name: String
        let value: String
        let isString: Bool
    }

    /// Decode a JSON object without discarding top-level member order or the
    /// original JSON spelling of non-string values. The official 0731 Python
    /// encoder iterates `json.loads(...).items()`, so this order determines
    /// the exact DSML history bytes and therefore the reusable KV prefix.
    private static func orderedJSONParameters(
        _ arguments: String
    ) -> [OrderedJSONParameter]? {
        guard let data = arguments.data(using: .utf8),
            (try? JSONSerialization.jsonObject(with: data)) is [String: Any]
        else { return nil }

        let text = arguments.trimmingCharacters(in: .whitespacesAndNewlines)
        guard text.first == "{", text.last == "}" else { return nil }

        var cursor = text.index(after: text.startIndex)
        let objectEnd = text.index(before: text.endIndex)
        var parameters: [OrderedJSONParameter] = []

        func skipWhitespace(_ index: inout String.Index) {
            while index < objectEnd, text[index].isWhitespace {
                index = text.index(after: index)
            }
        }

        func scanString(_ index: inout String.Index) -> Bool {
            guard index < text.endIndex, text[index] == "\"" else { return false }
            index = text.index(after: index)
            var escaped = false
            while index < text.endIndex {
                let character = text[index]
                index = text.index(after: index)
                if escaped {
                    escaped = false
                } else if character == "\\" {
                    escaped = true
                } else if character == "\"" {
                    return true
                }
            }
            return false
        }

        func scanValue(_ index: inout String.Index) -> Bool {
            guard index < objectEnd else { return false }
            if text[index] == "\"" {
                return scanString(&index)
            }
            if text[index] == "{" || text[index] == "[" {
                var stack: [Character] = []
                var inString = false
                var escaped = false
                while index < text.endIndex {
                    let character = text[index]
                    index = text.index(after: index)
                    if inString {
                        if escaped {
                            escaped = false
                        } else if character == "\\" {
                            escaped = true
                        } else if character == "\"" {
                            inString = false
                        }
                        continue
                    }
                    if character == "\"" {
                        inString = true
                    } else if character == "{" {
                        stack.append("}")
                    } else if character == "[" {
                        stack.append("]")
                    } else if character == stack.last {
                        stack.removeLast()
                        if stack.isEmpty { return true }
                    }
                }
                return false
            }
            while index < objectEnd, text[index] != "," {
                index = text.index(after: index)
            }
            return true
        }

        skipWhitespace(&cursor)
        if cursor == objectEnd { return [] }
        while cursor < objectEnd {
            let keyStart = cursor
            guard scanString(&cursor) else { return nil }
            let keyJSON = String(text[keyStart ..< cursor])
            guard let keyData = keyJSON.data(using: .utf8),
                let key = try? JSONDecoder().decode(String.self, from: keyData)
            else { return nil }

            skipWhitespace(&cursor)
            guard cursor < objectEnd, text[cursor] == ":" else { return nil }
            cursor = text.index(after: cursor)
            skipWhitespace(&cursor)

            let valueStart = cursor
            guard scanValue(&cursor) else { return nil }
            let rawValue = String(text[valueStart ..< cursor])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !rawValue.isEmpty else { return nil }

            let isString = rawValue.first == "\""
            let renderedValue: String
            if isString {
                guard let valueData = rawValue.data(using: .utf8),
                    let decoded = try? JSONDecoder().decode(String.self, from: valueData)
                else { return nil }
                renderedValue = decoded
            } else {
                renderedValue = rawValue
            }
            parameters.append(
                OrderedJSONParameter(name: key, value: renderedValue, isString: isString)
            )

            skipWhitespace(&cursor)
            if cursor == objectEnd { break }
            guard text[cursor] == "," else { return nil }
            cursor = text.index(after: cursor)
            skipWhitespace(&cursor)
        }
        return parameters
    }

    private static func renderToolCallInvoke(
        name: String,
        orderedParameters: [OrderedJSONParameter],
        family: DeepseekV4ChatFamily = .deepseekV4
    ) -> String {
        let d = DeepseekV4Tokens.dsml
        let parameterTag = family.dsmlParameterTagName
        let invokeTag = family.dsmlInvokeTagName
        let inner = orderedParameters.map { parameter in
            "<\(d)\(parameterTag) name=\"\(parameter.name)\" string=\"\(parameter.isString ? "true" : "false")\">\(parameter.value)</\(d)\(parameterTag)>"
        }.joined(separator: "\n")
        return
            "<\(d)\(invokeTag) name=\"\(name)\">\n\(inner)\n</\(d)\(invokeTag)>"
    }

    /// Render with the model's own argument order when it is known.
    ///
    /// The `params:` overload sorts, which is stable but is NOT what the model emitted; re-rendering
    /// a parsed call that way changes the bytes and breaks prefix reuse for every turn after it.
    /// `ToolCall.Function.argumentOrder` survives a serialized history, so a caller that has one
    /// can reproduce the original spelling. Names absent from `order` are appended in sorted order
    /// so an incomplete order can never silently drop a parameter.
    public static func renderToolCallInvoke(
        name: String, params: [String: Any], order: [String]?,
        family: DeepseekV4ChatFamily = .deepseekV4
    ) -> String {
        guard let order, !order.isEmpty else {
            return renderToolCallInvoke(name: name, params: params, family: family)
        }
        let ordered = order.filter { params[$0] != nil }
        let remainder = params.keys.filter { !ordered.contains($0) }.sorted()
        return renderToolCallInvoke(
            name: name,
            orderedParameters: (ordered + remainder).compactMap { key in
                guard let v = params[key] else { return nil }
                // Same value spelling as the sorted path below; only the ORDER differs.
                let isString = v is String
                let value: String
                if isString, let str = v as? String {
                    value = str
                } else {
                    value =
                        (try? String(
                            data: JSONSerialization.data(
                                withJSONObject: v,
                                options: [.fragmentsAllowed, .withoutEscapingSlashes]),
                            encoding: .utf8)) ?? "\(v)"
                }
                return OrderedJSONParameter(name: key, value: value, isString: isString)
            },
            family: family)
    }

    public static func renderToolCallInvoke(
        name: String,
        params: [String: Any],
        family: DeepseekV4ChatFamily = .deepseekV4
    ) -> String {
        let d = DeepseekV4Tokens.dsml
        let parameterTag = family.dsmlParameterTagName
        let invokeTag = family.dsmlInvokeTagName
        var paramBlocks: [String] = []
        // Encode each param — string params carry `string="true"` and
        // a raw value; all other JSON types serialize as JSON with
        // `string="false"`.
        //
        // Iterate in SORTED key order. `params` is an unordered Dictionary, so
        // `for (k, v) in params` emitted a different parameter order in every
        // process. The official encoder walks `json.loads(...).items()` and is
        // therefore stable, and the ordered overload above preserves the
        // original JSON spelling; this fallback path (reached from
        // `renderToolCallInvoke(name:arguments:)` when the arguments string
        // does not scan as an ordered JSON object) was the one place that did
        // not. Re-rendering a tool call in chat history with a random parameter
        // order changes the prompt bytes run to run, so the prefix-cache
        // boundary for every turn after a tool call stops matching across app
        // restarts. Deterministic order cannot recover the model's original
        // emission order, but it makes the bytes reproducible, which is what
        // cache reuse actually requires.
        for k in params.keys.sorted() {
            guard let v = params[k] else { continue }
            let isString = v is String
            let value: String
            if isString, let s = v as? String {
                value = s
            } else {
                value =
                    (try? String(
                        data: JSONSerialization.data(
                            withJSONObject: v, options: [.fragmentsAllowed, .withoutEscapingSlashes]
                        ),
                        encoding: .utf8)) ?? "\(v)"
            }
            let line =
                "<\(d)\(parameterTag) name=\"\(k)\" string=\"\(isString ? "true" : "false")\">\(value)</\(d)\(parameterTag)>"
            paramBlocks.append(line)
        }
        let inner = paramBlocks.joined(separator: "\n")
        return
            "<\(d)\(invokeTag) name=\"\(name)\">\n\(inner)\n</\(d)\(invokeTag)>"
    }

    // MARK: - Multi-turn helpers (drop_thinking, tool-merge)

    /// Strip `reasoning_content` from all assistant messages BEFORE the
    /// last user message. Keep user/system/tool/latest_reminder always.
    /// Mirrors `_drop_thinking_messages` in encoding_dsv4.py.
    static func dropThinkingMessages(
        _ messages: [Message],
        family: DeepseekV4ChatFamily = .deepseekV4
    ) -> [Message] {
        let lastUserIdx = findLastUserIndex(messages, family: family)
        let keepRoles: Set<MessageRole> = [.user, .system, .tool, .latestReminder]
        var result: [Message] = []
        for (idx, msg) in messages.enumerated() {
            if keepRoles.contains(msg.role) || idx >= lastUserIdx {
                result.append(msg)
            } else if msg.role == .assistant {
                var copy = msg
                copy.reasoningContent = nil
                result.append(copy)
            }
            // developer before last user: drop entirely.
        }
        return result
    }

    /// Merge `role:"tool"` messages into the preceding user message via
    /// `content_blocks`. DSV4 does NOT carry a standalone tool role.
    /// Mirrors `merge_tool_messages` in encoding_dsv4.py.
    static func mergeToolMessages(_ messages: [Message]) -> [Message] {
        var merged: [Message] = []
        for raw in messages {
            var m = raw
            switch m.role {
            case .tool:
                let block = MessageContentBlock.toolResult(
                    toolUseId: m.toolCallId ?? "", content: m.content ?? "")
                if var last = merged.last, last.role == .user,
                    var blocks = last.contentBlocks, !blocks.isEmpty
                {
                    blocks.append(block)
                    last.contentBlocks = blocks
                    merged[merged.count - 1] = last
                } else {
                    merged.append(
                        Message(role: .user, contentBlocks: [block]))
                }
            case .user:
                let textBlock = MessageContentBlock.text(m.content ?? "")
                if var last = merged.last,
                    last.role == .user,
                    var blocks = last.contentBlocks,
                    !blocks.isEmpty,
                    last.task == nil
                {
                    blocks.append(textBlock)
                    last.contentBlocks = blocks
                    last.task = m.task
                    last.tools = m.tools ?? last.tools
                    last.responseFormat = m.responseFormat ?? last.responseFormat
                    last.woEOS = m.woEOS
                    merged[merged.count - 1] = last
                } else {
                    m.contentBlocks = [textBlock]
                    merged.append(m)
                }
            default:
                merged.append(m)
            }
        }
        return merged
    }

    /// Sort `tool_result` blocks inside user messages by the order the
    /// calls appear in the preceding assistant turn. Python has the
    /// same behavior — needed because OpenAI-style tool responses can
    /// arrive out of order.
    static func sortToolResultsByCallOrder(_ messages: [Message]) -> [Message] {
        var lastCallOrder: [String: Int] = [:]
        var out: [Message] = []
        for raw in messages {
            var msg = raw
            if msg.role == .assistant, let tcs = msg.toolCalls, !tcs.isEmpty {
                lastCallOrder.removeAll()
                for (idx, tc) in tcs.enumerated() {
                    if let id = tc.id {
                        lastCallOrder[id] = idx
                    }
                }
            } else if msg.role == .user, var blocks = msg.contentBlocks, !blocks.isEmpty {
                let toolBlocks = blocks.compactMap { block -> (Int, MessageContentBlock)? in
                    if case .toolResult(let id, _) = block {
                        return (lastCallOrder[id] ?? Int.max, block)
                    }
                    return nil
                }
                if toolBlocks.count > 1 && !lastCallOrder.isEmpty {
                    let sorted = toolBlocks.sorted { $0.0 < $1.0 }.map { $0.1 }
                    var sortedIdx = 0
                    var newBlocks: [MessageContentBlock] = []
                    for block in blocks {
                        if case .toolResult = block {
                            newBlocks.append(sorted[sortedIdx])
                            sortedIdx += 1
                        } else {
                            newBlocks.append(block)
                        }
                    }
                    blocks = newBlocks
                    msg.contentBlocks = blocks
                }
            }
            out.append(msg)
        }
        return out
    }

    static func findLastUserIndex(
        _ messages: [Message],
        family: DeepseekV4ChatFamily = .deepseekV4
    ) -> Int {
        if family == .deepseekV41 {
            // encoding.py `find_last_user_index`: mid-conversation system
            // messages count as user turns, and a user message whose content
            // blocks are ONLY tool results does not start a new turn.
            for idx in (0..<messages.count).reversed() {
                let r = messages[idx].role
                guard r == .user || r == .developer || (r == .system && idx > 0) else {
                    continue
                }
                if let blocks = messages[idx].contentBlocks, !blocks.isEmpty,
                    !blocks.contains(where: {
                        if case .text = $0 { return true }
                        return false
                    })
                {
                    continue
                }
                return idx
            }
            return -1
        }
        for idx in (0..<messages.count).reversed() {
            let r = messages[idx].role
            if r == .user || r == .developer { return idx }
        }
        return -1
    }
}

// MARK: - DSV4-scoped message types
//
// Namespaced inside `DeepseekV4ChatEncoder` to avoid clashing with the
// existing top-level `Chat.Message` / `ToolCall` types in MLXLMCommon.
// The DSV4 encoder has extra fields (reasoningContent, contentBlocks,
// task, woEOS) that Chat.Message doesn't carry, so keeping them
// separate is cleaner than inflating the general Chat type.

extension DeepseekV4ChatEncoder {

    public enum MessageRole: String, Sendable, Codable, Hashable {
        case system
        case developer
        case user
        case assistant
        case tool
        case latestReminder = "latest_reminder"
    }

    public enum MessageContentBlock: Sendable, Hashable {
        case text(String)
        case toolResult(toolUseId: String, content: String)
    }

    public struct ToolCall: Sendable, Hashable {
        public let id: String?
        public let name: String
        public let arguments: String  // JSON string per OpenAI convention

        public init(id: String? = nil, name: String, arguments: String) {
            self.id = id
            self.name = name
            self.arguments = arguments
        }
    }

    public struct Message: Sendable {
        public var role: MessageRole
        public var content: String?
        public var contentBlocks: [MessageContentBlock]?
        public var reasoningContent: String?
        public var toolCalls: [ToolCall]?
        public var toolCallId: String?
        public var tools: [[String: any Sendable]]?
        public var responseFormat: [String: any Sendable]?
        public var task: String?
        public var woEOS: Bool

        public init(
            role: MessageRole,
            content: String? = nil,
            contentBlocks: [MessageContentBlock]? = nil,
            reasoningContent: String? = nil,
            toolCalls: [ToolCall]? = nil,
            toolCallId: String? = nil,
            tools: [[String: any Sendable]]? = nil,
            responseFormat: [String: any Sendable]? = nil,
            task: String? = nil,
            woEOS: Bool = false
        ) {
            self.role = role
            self.content = content
            self.contentBlocks = contentBlocks
            self.reasoningContent = reasoningContent
            self.toolCalls = toolCalls
            self.toolCallId = toolCallId
            self.tools = tools
            self.responseFormat = responseFormat
            self.task = task
            self.woEOS = woEOS
        }
    }
}

/// Short type aliases kept top-level so call sites inside this file
/// don't have to spell out the DSV4 namespace on every reference.
/// External callers always go through `DeepseekV4ChatEncoder.Message`
/// etc. — these aliases are private to the file.
typealias DSV4Message = DeepseekV4ChatEncoder.Message
typealias DSV4MessageRole = DeepseekV4ChatEncoder.MessageRole
typealias DSV4ContentBlock = DeepseekV4ChatEncoder.MessageContentBlock
typealias DSV4ToolCall = DeepseekV4ChatEncoder.ToolCall

// MARK: - JSON serialization helper

private extension Dictionary where Key == String, Value == any Sendable {
    func jsonSerialized() -> String {
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: self, options: [.withoutEscapingSlashes]),
            let object = try? JSONSerialization.jsonObject(with: data)
        else { return "{}" }
        return Self.pythonStyleJSON(object)
    }

    /// Match `json.dumps(...)` as used by the official 0731
    /// `encoding_dsv4.py` tool-schema renderer. Foundation's compact JSON
    /// removes the spaces after `:`/`,` and emits hash-order dictionaries;
    /// both change the model's prompt tokens even though the JSON is
    /// semantically equivalent. DSV4's low-bit affine bundles are
    /// particularly sensitive to exact identifier-copy prompts, so keep a
    /// deterministic, native-shaped representation here.
    static func pythonStyleJSON(_ value: Any) -> String {
        if let dictionary = value as? [String: Any] {
            let keys = orderedJSONKeys(in: dictionary)
            return "{" + keys.map { key in
                "\(pythonStyleJSON(key)): \(pythonStyleJSON(dictionary[key]!))"
            }.joined(separator: ", ") + "}"
        }
        if let array = value as? [Any] {
            return "[" + array.map(Self.pythonStyleJSON).joined(separator: ", ") + "]"
        }
        guard let scalar = try? JSONSerialization.data(
            withJSONObject: value,
            options: [.fragmentsAllowed, .withoutEscapingSlashes]
        ) else { return "null" }
        return String(data: scalar, encoding: .utf8) ?? "null"
    }

    static func orderedJSONKeys(in dictionary: [String: Any]) -> [String] {
        let preferred: [String]
        if dictionary["name"] != nil, dictionary["parameters"] != nil {
            preferred = ["name", "description", "parameters", "strict"]
        } else if dictionary["type"] != nil, dictionary["properties"] != nil {
            preferred = [
                "type", "properties", "required", "additionalProperties",
                "description", "title", "$defs",
            ]
        } else if dictionary["type"] != nil {
            preferred = [
                "type", "enum", "items", "description", "default", "examples",
                "minimum", "maximum", "minLength", "maxLength", "pattern",
            ]
        } else {
            preferred = []
        }
        var rank: [String: Int] = [:]
        for (index, key) in preferred.enumerated() {
            rank[key] = index
        }
        return dictionary.keys.sorted { lhs, rhs in
            let leftRank = rank[lhs] ?? Int.max
            let rightRank = rank[rhs] ?? Int.max
            return leftRank == rightRank ? lhs < rhs : leftRank < rightRank
        }
    }
}
