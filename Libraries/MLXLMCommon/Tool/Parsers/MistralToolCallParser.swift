// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for the Mistral tekken tool-call formats.
///
/// Mistral has shipped two on-the-wire encodings, both opened by the
/// `[TOOL_CALLS]` special token (ID 9) and ended at EOS (no end tag):
///
/// - V7 / V11 (Mistral-Small-3.1/3.2 2503/2506, Pixtral): a JSON array of
///   call objects — `[TOOL_CALLS][{"name": "get_weather", "arguments": {...}}]`.
///   These tokenizers do NOT contain an `[ARGS]` token.
/// - V13 (Mistral-3 2512, Devstral 2): `[TOOL_CALLS]name[ARGS]{json}`, with an
///   optional `[CALL_ID]` between the name and `[ARGS]` (the V11 variant).
///
/// This parser accepts both: it uses `[ARGS]` when present, otherwise decodes
/// the remainder as a JSON array (or single object) of `{name, arguments}`.
/// Since stop tokens are intercepted at the token-ID level before
/// detokenization, the buffered tool body is handed to `parseEOS()` at the end
/// of generation.
///
/// Examples:
/// - `[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "Tokyo"}}]`
/// - `[TOOL_CALLS]get_weather[ARGS]{"location": "Tokyo"}`
/// - `[TOOL_CALLS]fn1[ARGS]{...}[TOOL_CALLS]fn2[ARGS]{...}` (multiple calls)
public struct MistralToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "[TOOL_CALLS]"
    public let endTag: String? = nil

    /// V7/V11 checkpoints drop the `[TOOL_CALLS]` token on tool turns whose
    /// history contains images, emitting the call array as plain text
    /// (`[{"name": "get_weather", "arguments": {...}}]`). Let the processor
    /// buffer that exact shape for registered tools.
    public let supportsBareJSONArrayToolFallback = true

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        var text = content.trimmingCharacters(in: .whitespacesAndNewlines)

        // Strip wrapper tags only when they appear at boundaries.
        // This keeps literal tag strings inside argument values intact.
        if let start = startTag, text.hasPrefix(start) {
            text = String(text.dropFirst(start.count))
            text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        // V13 / V11: `[name][CALL_ID?][ARGS]{json}`
        if let argsRange = text.range(of: "[ARGS]") {
            var namePart = String(text[..<argsRange.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let argsPart = String(text[argsRange.upperBound...])
                .trimmingCharacters(in: .whitespacesAndNewlines)

            // Handle optional [CALL_ID] between name and [ARGS]
            if let callIdRange = namePart.range(of: "[CALL_ID]") {
                namePart = String(namePart[..<callIdRange.lowerBound])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }

            guard !namePart.isEmpty else { return nil }
            guard let argsDict = tryParseJSON(argsPart) as? [String: any Sendable] else {
                return nil
            }
            return ToolCall(function: ToolCall.Function(name: namePart, arguments: argsDict))
        }

        // V7 / V11 (2503/2506): JSON array or single object of {name, arguments}.
        return Self.firstToolCall(fromJSON: text)
    }

    public func parseEOS(
        _ toolCallBuffer: String, tools: [[String: any Sendable]]?
    ) -> [ToolCall] {
        // Each `[TOOL_CALLS]` segment is either a single `name[ARGS]{json}` call
        // (V13) or a JSON array/object that may itself carry multiple calls (V7).
        let segments =
            toolCallBuffer
            .components(separatedBy: startTag ?? "[TOOL_CALLS]")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        var calls: [ToolCall] = []
        for segment in segments {
            if segment.range(of: "[ARGS]") != nil {
                if let call = parse(content: segment, tools: tools) {
                    calls.append(call)
                }
            } else if let parsed = tryParseJSON(segment) as? [Any] {
                for element in parsed {
                    if let call = Self.toolCall(fromObject: element) {
                        calls.append(call)
                    }
                }
            } else if let call = Self.firstToolCall(fromJSON: segment) {
                calls.append(call)
            }
        }
        return calls
    }

    /// Decode a JSON array (first element) or single object into a `ToolCall`.
    private static func firstToolCall(fromJSON text: String) -> ToolCall? {
        let parsed = tryParseJSON(text)
        if let array = parsed as? [Any] {
            return array.lazy.compactMap { toolCall(fromObject: $0) }.first
        }
        return toolCall(fromObject: parsed)
    }

    /// Map a `{"name": ..., "arguments": ...}` (or `parameters`) object to a
    /// `ToolCall`. Accepts string-encoded arguments as well as objects.
    private static func toolCall(fromObject object: Any?) -> ToolCall? {
        guard let dict = object as? [String: any Sendable] else { return nil }
        // Some encodings nest under `function`.
        if let fn = dict["function"] as? [String: any Sendable] {
            return toolCall(fromObject: fn)
        }
        guard let name = (dict["name"] as? String), !name.isEmpty else { return nil }
        let rawArgs = dict["arguments"] ?? dict["parameters"]
        let args: [String: any Sendable]
        if let mapping = rawArgs as? [String: any Sendable] {
            args = mapping
        } else if let string = rawArgs as? String,
            let mapping = tryParseJSON(string) as? [String: any Sendable]
        {
            args = mapping
        } else {
            args = [:]
        }
        return ToolCall(function: ToolCall.Function(name: name, arguments: args))
    }
}
