// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for Gemma format: call:name{key:value,k:<escape>str<escape>}
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/function_gemma.py
public struct GemmaFunctionParser: ToolCallParser, Sendable {
    public let startTag: String?
    public let endTag: String?
    public let supportsBareCallToolFallback = true

    private let escapeMarker: String

    /// Initialize with Gemma 3 tags (default)
    public init() {
        self.startTag = "<start_function_call>"
        self.endTag = "<end_function_call>"
        self.escapeMarker = "<escape>"
    }

    /// Initialize with custom tags (for Gemma 4 which uses `<|tool_call>` / `<tool_call|>`)
    /// Default escape marker is Gemma-4's `<|"|>` (three characters: `<`, `|`, `"`, `|`, `>`).
    /// Previous default contained a spurious backslash (`<|"\|>`) that never appeared in
    /// real Gemma-4 output; callers using this init directly without the `.gemma4` factory
    /// would silently fail to decode string values. Factory path was always correct.
    public init(startTag: String, endTag: String, escapeMarker: String = "<|\"|>") {
        self.startTag = startTag
        self.endTag = endTag
        self.escapeMarker = escapeMarker
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        // Strip tags if present
        var text = content
        if let start = startTag {
            text = text.replacingOccurrences(of: start, with: "")
        }
        if let end = endTag {
            text = text.replacingOccurrences(of: end, with: "")
        }

        // Pattern: call:(\w+)\{(.*?)\}
        // Find "call:" followed by function name and arguments in braces
        guard let callRange = text.range(of: "call:") else { return nil }

        let remaining = String(text[callRange.upperBound...])

        // Extract function name. Native format is `call:name{...}`; live rows
        // can drift to a paren-wrapped form `call:name({...})`. Stop the name
        // at the first `{` (native) or `(` (drift) so the trailing paren never
        // becomes part of the function name. The argument span below still
        // reads the inner `{...}`, so JSON-in-parens parses identically.
        guard let braceStart = remaining.firstIndex(of: "{") else { return nil }
        let nameEnd = remaining.firstIndex(where: { $0 == "{" || $0 == "(" }) ?? braceStart
        let funcName = String(remaining[..<nameEnd])
            .trimmingCharacters(in: .whitespacesAndNewlines)

        guard !funcName.isEmpty else { return nil }

        // Extract arguments string (everything between { and })
        guard let braceEnd = remaining.lastIndex(of: "}") else { return nil }
        var argsStr = String(remaining[remaining.index(after: braceStart) ..< braceEnd])

        var arguments: [String: any Sendable] = [:]

        // Parse key:value pairs
        while !argsStr.isEmpty {
            argsStr = argsStr.trimmingCharacters(in: .whitespacesAndNewlines)
            if argsStr.isEmpty { break }

            // Find the key (everything before :)
            guard let colonIdx = argsStr.firstIndex(of: ":") else { break }
            let key = normalizeArgumentKey(
                String(argsStr[..<colonIdx])
                    .trimmingCharacters(in: .whitespacesAndNewlines))
            argsStr = String(argsStr[argsStr.index(after: colonIdx)...])

            guard let value = parseValue(from: &argsStr) else { break }
            arguments[key] = value

            argsStr = argsStr.trimmingCharacters(in: .whitespacesAndNewlines)
            if argsStr.hasPrefix(",") {
                argsStr = String(argsStr.dropFirst())
            }
        }

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }

    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        if let startTag {
            let tagged = toolCallBuffer
                .components(separatedBy: startTag)
                .filter { !$0.isEmpty }
                .compactMap { parse(content: $0, tools: tools) }
            if !tagged.isEmpty {
                return tagged
            }
        }

        // Some Gemma4 VLM rows emit the canonical call body without the
        // `<|tool_call>` wrapper at EOS, e.g. `call:get_weather{...}`.
        // Treat only an actual call body as a tool call; ordinary visible text
        // remains visible through ToolCallProcessor's normal streaming path.
        guard toolCallBuffer.range(of: "call:") != nil,
              let bare = parse(content: toolCallBuffer, tools: tools)
        else {
            return []
        }
        return [bare]
    }

    private func parseValue(from text: inout String) -> (any Sendable)? {
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.hasPrefix(escapeMarker) {
            text = String(text.dropFirst(escapeMarker.count))
            guard let endEscape = text.range(of: escapeMarker) else { return nil }
            let value = String(text[..<endEscape.lowerBound])
            text = String(text[endEscape.upperBound...])
            return decodeEscapedStringValue(value)
        }

        if text.hasPrefix("[") {
            return parseArray(from: &text)
        }

        let value = takeRawValue(from: &text, terminators: [","])
        return decodeRawValue(value)
    }

    private func parseArray(from text: inout String) -> [any Sendable]? {
        guard text.hasPrefix("[") else { return nil }
        text = String(text.dropFirst())
        var values: [any Sendable] = []

        while true {
            text = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if text.hasPrefix("]") {
                text = String(text.dropFirst())
                return values
            }

            if text.isEmpty { return nil }

            if let value = parseValue(from: &text) {
                values.append(value)
            } else {
                return nil
            }

            text = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if text.hasPrefix(",") {
                text = String(text.dropFirst())
                continue
            }
            if text.hasPrefix("]") {
                text = String(text.dropFirst())
                return values
            }
            return nil
        }
    }

    private func takeRawValue(from text: inout String, terminators: Set<Character>) -> String {
        var index = text.startIndex
        while index < text.endIndex {
            let character = text[index]
            if terminators.contains(character) || character == "]" {
                break
            }
            index = text.index(after: index)
        }

        let value = String(text[..<index]).trimmingCharacters(in: .whitespacesAndNewlines)
        text = String(text[index...])
        return value
    }

    private func decodeRawValue(_ value: String) -> any Sendable {
        guard !value.isEmpty else { return "" }
        if value.hasPrefix("\""),
           value.hasSuffix("\"")
        {
            if let data = value.data(using: .utf8),
               let decoded = try? JSONDecoder().decode(String.self, from: data)
            {
                return decoded
            }
            return decodeQuotedStringLiteral(value)
        }
        if value.hasPrefix("'"),
           value.hasSuffix("'")
        {
            return String(value.dropFirst().dropLast())
        }
        if let data = value.data(using: .utf8),
            let json = deserializeJSON(data)
        {
            return json
        }
        return value
    }

    private func decodeEscapedStringValue(_ value: String) -> String {
        if value.hasPrefix("\""),
           value.hasSuffix("\"")
        {
            if let data = value.data(using: .utf8),
                let decoded = try? JSONDecoder().decode(String.self, from: data)
            {
                return decoded
            }
            return decodeQuotedStringLiteral(value)
        }
        return decodeBackslashEscapedString(value)
    }

    private func normalizeArgumentKey(_ key: String) -> String {
        guard key.count >= 2 else { return key }
        if key.hasPrefix("\""), key.hasSuffix("\"") {
            if let data = key.data(using: .utf8),
                let decoded = try? JSONDecoder().decode(String.self, from: data)
            {
                return decoded
            }
            return decodeQuotedStringLiteral(key)
        }
        if key.hasPrefix("'"), key.hasSuffix("'") {
            return String(key.dropFirst().dropLast())
        }
        return key
    }

    private func decodeBackslashEscapedString(_ value: String) -> String {
        var inner = value
        inner = inner.replacingOccurrences(of: #"\\n"#, with: #"\n"#)
        inner = inner.replacingOccurrences(of: #"\\t"#, with: #"\t"#)
        inner = inner.replacingOccurrences(of: #"\\r"#, with: #"\r"#)
        inner = inner.replacingOccurrences(of: #"\""#, with: #"""#)
        inner = inner.replacingOccurrences(of: #"\n"#, with: "\n")
        inner = inner.replacingOccurrences(of: #"\t"#, with: "\t")
        inner = inner.replacingOccurrences(of: #"\r"#, with: "\r")
        inner = inner.replacingOccurrences(of: #"\\"#, with: #"\"#)
        return inner
    }

    private func decodeQuotedStringLiteral(_ value: String) -> String {
        decodeBackslashEscapedString(String(value.dropFirst().dropLast()))
    }
}
