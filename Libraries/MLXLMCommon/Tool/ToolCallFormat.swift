// Copyright © 2025 Apple Inc.

import Foundation

// MARK: - ToolCallParser Protocol

/// Protocol for parsing tool call content from model output.
///
/// Different models use different formats for tool calls. This protocol provides
/// a common interface for parsing tool calls from model output text.
///
/// Reference: https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/tool_parsers
public protocol ToolCallParser: Sendable {
    /// The start tag that indicates a tool call is beginning.
    /// Returns `nil` for inline formats that don't use wrapper tags.
    var startTag: String? { get }

    /// The end tag that indicates a tool call has ended.
    /// Returns `nil` for inline formats that don't use wrapper tags.
    var endTag: String? { get }

    /// Parse the content into a `ToolCall`.
    /// - Parameters:
    ///   - content: The text content to parse (may include tags)
    ///   - tools: Optional tool schemas for type-aware parsing
    /// - Returns: A `ToolCall` if parsing succeeds, `nil` otherwise
    func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall?

    /// Parse remaining buffered content at end-of-sequence.
    ///
    /// Called when generation ends to extract any tool calls still in the buffer.
    /// The default implementation splits on `startTag` (if present) and parses
    /// each segment individually.
    func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall]

    /// Return whether the current tagged buffer can still become a valid tool
    /// call for this format. Tagged parsers default to permissive collection;
    /// stricter formats can reject impossible prefixes so literal tag-looking
    /// prose is surfaced instead of being buffered forever.
    func isValidPartialContent(_ toolCallBuffer: String) -> Bool
}

extension ToolCallParser {
    public func isValidPartialContent(_ toolCallBuffer: String) -> Bool {
        true
    }

    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        if let startTag {
            return
                toolCallBuffer
                .components(separatedBy: startTag)
                .filter { !$0.isEmpty }
                .compactMap { parse(content: $0, tools: tools) }
        } else {
            guard let toolCall = parse(content: toolCallBuffer, tools: tools) else {
                return []
            }
            return [toolCall]
        }
    }
}

// MARK: - ToolCallFormat Enum

/// Supported tool call formats for different language models.
///
/// This enum defines the various tool call formats used by different LLM families.
/// Each format has its own syntax for encoding function names and arguments.
///
/// The raw string values can be used for JSON serialization or CLI parameters.
///
/// Reference: https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/tool_parsers
public enum ToolCallFormat: String, Sendable, Codable, CaseIterable {
    /// Default JSON format used by Llama, Qwen, and most models.
    /// Example: `<tool_call>{"name": "func", "arguments": {...}}</tool_call>`
    case json

    /// LFM2/LFM2.5 Pythonic format with model-specific tags.
    /// Example: `<|tool_call_start|>[func(arg='value')]<|tool_call_end|>`
    case lfm2

    /// XML function format used by Nemotron, Qwen3 Coder, Qwen3.5, and similar models.
    /// Example: `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`
    case xmlFunction = "xml_function"

    /// GLM4 format with arg_key/arg_value tags.
    /// Example: `func<arg_key>k</arg_key><arg_value>v</arg_value>`
    case glm4

    /// Gemma 3 function call format.
    /// Example: `<start_function_call>call:name{key:<escape>value<escape>}<end_function_call>`
    case gemma

    /// Gemma 4 function call format (different tags from Gemma 3).
    /// Example: `<|tool_call>call:name{key:<|"|>value<|"|>}<tool_call|>`
    case gemma4

    /// Kimi K2 format with functions prefix.
    /// Example: `functions.name:0<|tool_call_argument_begin|>{"key": "value"}`
    case kimiK2 = "kimi_k2"

    /// MiniMax M2 format with invoke/parameter tags.
    /// Example: `<invoke name="f"><parameter name="k">v</parameter></invoke>`
    case minimaxM2 = "minimax_m2"

    /// Mistral V11+ format with [TOOL_CALLS] and [ARGS] delimiters.
    /// Example: `[TOOL_CALLS]get_weather [ARGS]{"location": "Tokyo"}`
    case mistral

    /// Llama 3 inline JSON format.
    /// Example: `<|python_tag|>{ "name": "func", "parameters": {...} }`
    case llama3

    /// DSML (DeepSeek Markup Language) used by DeepSeek-V4-Flash /
    /// -Pro per jang/research/DSV-FAMILY-RUNTIME-GUIDE.md §24.
    /// Example: `<｜DSML｜tool_calls><｜DSML｜invoke name="f"><｜DSML｜parameter name="k" string="true">v</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>`
    /// (markers use fullwidth vertical bar U+FF5C, not ASCII `|`).
    case dsml

    /// ZAYA XML function format. The inner function/parameter syntax is the
    /// same as Qwen/Nemotron XML function calls, but Zyphra wraps calls in
    /// `<zyphra_tool_call>...</zyphra_tool_call>`.
    case zayaXml = "zaya_xml"

    /// Tencent Hunyuan / Hy3 XML-like tool-call wrapper.
    /// Example:
    /// `<tool_calls><tool_call>f<tool_sep><arg_key>k</arg_key><arg_value>v</arg_value></tool_call></tool_calls>`.
    case hunyuan

    // MARK: - Factory Methods

    /// Create the appropriate parser for this format.
    /// - Returns: A parser instance configured for this format
    public func createParser() -> any ToolCallParser {
        switch self {
        case .json:
            return JSONToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
        case .lfm2:
            return PythonicToolCallParser(
                startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        case .xmlFunction:
            return XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
        case .glm4:
            return GLM4ToolCallParser()
        case .gemma:
            return GemmaFunctionParser()
        case .gemma4:
            return GemmaFunctionParser(
                startTag: "<|tool_call>", endTag: "<tool_call|>", escapeMarker: "<|\"|>")
        case .kimiK2:
            return KimiK2ToolCallParser()
        case .minimaxM2:
            return MiniMaxM2ToolCallParser()
        case .mistral:
            return MistralToolCallParser()
        case .llama3:
            return Llama3ToolCallParser()
        case .dsml:
            return DSMLToolCallParser()
        case .zayaXml:
            return XMLFunctionParser(
                startTag: "<zyphra_tool_call>",
                endTag: "</zyphra_tool_call>")
        case .hunyuan:
            return HunyuanToolCallParser()
        }
    }

    /// Infer the tool call format based on model type from config.json.
    ///
    /// This method maps known model types to their corresponding tool call formats,
    /// enabling automatic format detection when loading models.
    ///
    /// - Parameters:
    ///   - modelType: The `model_type` value from config.json
    ///   - configData: The raw config.json data for inspecting secondary signals
    ///     (e.g. `rope_scaling` / `vocab_size` for Llama 3 vs Llama 2).
    /// - Returns: The appropriate `ToolCallFormat`, or `nil` to use the default format
    public static func infer(from modelType: String, configData: Data? = nil) -> ToolCallFormat? {
        let type = modelType.lowercased()

        // Llama family (need secondary signal for Llama 3 vs 1/2).
        // Kept byte-compatible with upstream ml-explore/mlx-swift-lm.
        if type == "llama" {
            guard let data = configData,
                let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { return nil }

            // Secondary signal 1: vocab_size >= 128000 (Llama 3 uses 128256, Llama 2 uses 32000)
            if let vocabSize = json["vocab_size"] as? Int, vocabSize >= 128000 {
                return .llama3
            }

            // Secondary signal 2: rope_scaling with rope_type == "llama3"
            if let ropeScaling = json["rope_scaling"] as? [String: Any],
                let ropeType = ropeScaling["rope_type"] as? String,
                ropeType == "llama3"
            {
                return .llama3
            }

            return nil
        }

        // LFM2 family (lfm2, lfm2_moe, lfm2_5, lfm25, etc.)
        if type.hasPrefix("lfm2") {
            return .lfm2
        }

        // GLM/GLM-style families (glm4, glm4_moe, glm5, glm47, etc.).
        if type.hasPrefix("glm4")
            || type.hasPrefix("glm5")
            || type.hasPrefix("glm47")
        {
            return .glm4
        }

        // Ling/Bailing hybrid bundles stamp `tool_parser = "deepseek"` in
        // JANG metadata, but non-JANG/config-only fallbacks still need the
        // same GLM-style arg_key/arg_value parser instead of default JSON.
        if type.hasPrefix("bailing") {
            return .glm4
        }

        // Gemma family
        if type.hasPrefix("gemma4") {
            return .gemma4
        }
        if type.hasPrefix("gemma3") || type == "gemma" {
            return .gemma
        }

        // MiniMax family (minimax, minimax_m2)
        if type.hasPrefix("minimax") {
            return .minimaxM2
        }

        // Nemotron family (nemotron_h, etc.)
        if type.hasPrefix("nemotron") {
            return .xmlFunction
        }

        // Qwen3.5 family (qwen3_5, qwen3_5_moe, etc.)
        if type.hasPrefix("qwen3_5") {
            return .xmlFunction
        }

        // Qwen3.6 / Qwen3-VL use the same XML-function tool envelope as the
        // Qwen3.5 runtime family. Keep this as a model_type fallback for
        // source/config-only bundles; JANG stamps still take precedence.
        if type.hasPrefix("qwen3_6") || type.hasPrefix("qwen3_vl") {
            return .xmlFunction
        }

        // Qwen3-Next family (qwen3_next, etc.)
        if type.hasPrefix("qwen3_next") {
            return .xmlFunction
        }

        // Mistral3 family (mistral3, mistral3_text, etc.)
        if type.hasPrefix("mistral3") {
            return .mistral
        }

        // Ministral3 (Mistral 3.5 inner text_config.model_type). When a
        // bundle exposes the inner type at the outer level (rare but
        // possible for text-only Ministral3 LLM bundles), match it
        // here so tool calling routes correctly. The outer `mistral3`
        // wrapper case is already handled above.
        if type.hasPrefix("ministral3") {
            return .mistral
        }

        // Mistral 4 and Pixtral-family bundles share the Mistral tool
        // protocol. Some converted bundles expose the inner text decoder as
        // `mistral4` directly rather than through a `mistral3` VLM wrapper.
        if type.hasPrefix("mistral4")
            || type.hasPrefix("mistral_4")
            || type.hasPrefix("pixtral")
        {
            return .mistral
        }

        // Laguna (Poolside agentic-coding MoE). `laguna_glm_thinking_v5/
        // chat_template.jinja` uses GLM-family function-calling tags.
        // Matches the same parser as glm4_moe / glm5 / deepseek (V3
        // family) which all share the GLM-style tool format.
        if type.hasPrefix("laguna") {
            return .glm4
        }

        // Kimi family (kimi_k2, kimi_k15, etc.). JANG converters stamp
        // `capabilities.toolParser = "kimi_k2"`; non-JANG bundles fall
        // through to this model_type sniff.
        if type.hasPrefix("kimi") {
            return .kimiK2
        }

        // DeepSeek-V4 — `DSML` markup format. Per
        // `jang/research/DSV-FAMILY-RUNTIME-GUIDE.md` §24 the
        // `jang_config.chat.tool_calling.parser = "dsml"` stamp is
        // authoritative via `fromCapabilityName` below. This
        // model_type sniff catches non-JANG DSV4 bundles too.
        //
        // NOTE: intentionally narrower than `"deepseek"` prefix —
        // DSV3 / DSV3.2 / Kimi K2.x use the Kimi/GLM4-style tool
        // format, not DSML. We only trigger DSML on explicit `_v4`.
        if type.hasPrefix("deepseek_v4") {
            return .dsml
        }

        // ZAYA uses Qwen-style XML function bodies with Zyphra-specific
        // wrapper tags. JANG stamps this as `zaya_xml`; this fallback covers
        // plain/non-JANG ZAYA configs.
        if type.hasPrefix("zaya") || type.hasPrefix("zyphra") {
            return .zayaXml
        }

        // Tencent Hunyuan v3 / Hy3 uses its own XML-like tool wrapper.
        if type == "hy_v3" || type == "hy-v3" || type.hasPrefix("hy3") {
            return .hunyuan
        }

        return nil
    }

    /// Resolve a `JangCapabilities.toolParser` value into a canonical
    /// `ToolCallFormat`.
    ///
    /// The JANG converter stamps short, family-style names (`qwen`,
    /// `minimax`, `glm47`, `deepseek`, `nemotron`, `gemma4`, `mistral`)
    /// rather than vmlx's enum raw values (`xml_function`, `minimax_m2`,
    /// `glm4`, ...). This factory accepts both spellings plus the
    /// vLLM-ecosystem standard `qwen3_coder`.
    ///
    /// Returns `nil` when the name is unknown or empty — callers should
    /// fall back to `infer(from: model_type)`.
    public static func fromCapabilityName(_ name: String?) -> ToolCallFormat? {
        guard let name, !name.isEmpty else { return nil }
        let n = name.lowercased()
        let normalized = n.replacingOccurrences(of: "-", with: "_")

        // Direct rawValue match first (e.g. "xml_function", "minimax_m2").
        if let direct = ToolCallFormat(rawValue: n) {
            return direct
        }

        if normalized.hasPrefix("mistral4")
            || normalized.hasPrefix("mistral_4")
            || normalized.hasPrefix("mistral_small_4")
            || normalized.hasPrefix("mistral_large_4")
            || normalized.hasPrefix("pixtral")
        {
            return .mistral
        }

        // DSV4 must resolve before the generic DeepSeek/GLM parser family.
        // V3-style DeepSeek aliases use the GLM arg_key/arg_value format, but
        // V4 Flash/Pro uses DSML.
        if normalized.hasPrefix("deepseek_v4")
            || normalized.hasPrefix("deepseekv4")
        {
            return .dsml
        }

        // Family aliases with minor-version suffixes. Capability metadata is
        // often stamped at the family level (`glm5_air`, `glm4_moe_lite`,
        // `deepseek_v3`, `laguna_glm_thinking_v5`) rather than the exact enum
        // raw value, and all of these non-DSV4 aliases use the
        // GLM/DeepSeek arg_key/arg_value
        // parser.
        if n.hasPrefix("glm4_")
            || n.hasPrefix("glm5")
            || n.hasPrefix("glm47")
            || n.hasPrefix("deepseek")
            || n.hasPrefix("laguna")
        {
            return .glm4
        }

        if normalized.hasPrefix("qwen3_vl")
            || normalized.hasPrefix("qwen3_5_vl")
            || normalized.hasPrefix("qwen3_6_vl")
        {
            return .xmlFunction
        }

        switch n {
        // Qwen 3.5 / 3.6 family — XML-style <tool_call>…</tool_call>
        // (vLLM ecosystem name `qwen3_coder` aliased here).
        case "qwen", "qwen3", "qwen3_5", "qwen35", "qwen3_6", "qwen36",
            "qwen3_coder":
            return .xmlFunction
        // MiniMax — JANG converter stamps `minimax`; older artifacts use
        // the canonical `minimax_m2`. Future M2.5 variants use
        // `minimax_m2_5` per the converter.
        case "minimax", "minimax_m2_5":
            return .minimaxM2
        // GLM 4.x / 5 / DeepSeek tool format (arg_key / arg_value tags).
        // `glm4` is also the canonical rawValue and already matches via
        // the direct lookup above, but is listed here for parity with
        // `glm4_moe` / `glm47` family aliases.
        case "glm4", "glm47", "glm5", "glm4_moe", "deepseek",
            "laguna", "laguna_xs", "laguna_s":
            return .glm4
        // Nemotron-H / Cascade — same XML-style envelope as Qwen3 Coder.
        // Our `XMLFunctionParser` handles `<tool_call><function=name>…`
        // which Nemotron's variant matches; if a future Nemotron release
        // uses Hermes-style `<TOOLCALL>` tags, a dedicated enum case
        // should be added here.
        case "nemotron", "nemotron_h":
            return .xmlFunction
        // Gemma — JANG stamps `gemma4`; the `gemma` short form maps to
        // legacy Gemma 3 format and is included for forward compatibility
        // with older stamps. Both produce `<|tool_call>…<tool_call|>`
        // style envelopes via `GemmaFunctionParser`.
        case "gemma":
            return .gemma
        case "gemma4":
            return .gemma4
        // Mistral 4 — `[TOOL_CALLS] … [ARGS] …` JSON delimiters.
        case "mistral", "mistral4":
            return .mistral
        // LFM2 — pythonic `[func(arg='v')]` between
        // `<|tool_call_start|>` / `<|tool_call_end|>`.
        case "lfm2", "lfm2_5":
            return .lfm2
        // KimiK2 — `functions.name:0<|tool_call_argument_begin|>{…}`.
        case "kimi", "kimik2", "kimi_k2":
            return .kimiK2
        // DSV4 DSML — authoritative stamp from
        // `jang_config.chat.tool_calling.parser`. `deepseek_v4`
        // alias catches bundles that stamp the model_family rather
        // than the parser.
        case "dsml", "deepseek_v4", "deepseekv4":
            return .dsml
        // ZAYA / Zyphra XML wrapper around the standard XML function body.
        case "zaya", "zaya_xml", "zyphra", "zyphra_xml":
            return .zayaXml
        // Tencent Hunyuan / Hy3 parser aliases. JANG stamps "hunyuan";
        // vLLM ecosystem examples use "hy_v3"; SGLang uses "hunyuan".
        case "hunyuan", "tencent", "hy3", "hy_v3", "hy-v3":
            return .hunyuan
        default:
            return nil
        }
    }
}
