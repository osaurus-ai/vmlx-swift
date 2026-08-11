// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for GLM4 format: func<arg_key>k</arg_key><arg_value>v</arg_value>
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/glm47.py
public struct GLM4ToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<tool_call>"
    public let endTag: String? = "</tool_call>"

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        // Strip tags if present
        var text = content
        if let start = startTag {
            text = text.replacingOccurrences(of: start, with: "")
        }
        if let end = endTag {
            text = text.replacingOccurrences(of: end, with: "")
        }
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)

        // Extract function name (everything before first <arg_key>).
        //
        // A tool that declares no parameters is invoked with the bare name —
        // `<tool_call>list_mailboxes</tool_call>` — so a missing `<arg_key>`
        // is a zero-argument call, not a malformed one. Treating it as a
        // parse failure dropped the call silently: the surface saw a turn with
        // no text and no tool work, nudged the model, got the same correct
        // call again, and after the retry budget told the user the model had
        // "returned empty output after tool execution".
        //
        // With no `<arg_key>` to bound it, the name is the entire body, so it
        // has to earn acceptance: prose wrapped in `<tool_call>` would
        // otherwise be promoted to a call named after the prose. Calls that do
        // carry arguments keep the original, laxer rule — their name is
        // already delimited, and tightening it here would reject spellings
        // that parse today.
        let nameEnd = text.range(of: "<arg_key>")?.lowerBound
        let funcName = String(text[..<(nameEnd ?? text.endIndex)])
            .trimmingCharacters(in: .whitespacesAndNewlines)

        if nameEnd == nil {
            guard Self.isFunctionNameShaped(funcName) else { return nil }
        } else {
            guard !funcName.isEmpty else { return nil }
        }

        var arguments: [String: any Sendable] = [:]

        // Find all arg_key/arg_value pairs
        var searchRange = text.startIndex ..< text.endIndex
        while let keyStart = text.range(of: "<arg_key>", range: searchRange) {
            // Find </arg_key>
            guard
                let keyEnd = text.range(
                    of: "</arg_key>", range: keyStart.upperBound ..< text.endIndex)
            else { break }

            let key = String(text[keyStart.upperBound ..< keyEnd.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)

            // Find <arg_value> after </arg_key>
            guard
                let valueStart = text.range(
                    of: "<arg_value>", range: keyEnd.upperBound ..< text.endIndex)
            else { break }

            // Find </arg_value>
            guard
                let valueEnd = text.range(
                    of: "</arg_value>", range: valueStart.upperBound ..< text.endIndex)
            else { break }

            let value = String(text[valueStart.upperBound ..< valueEnd.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)

            // GLM4: deserialize if NOT a string type in schema
            if !isStringType(funcName: funcName, argName: key, tools: tools) {
                arguments[key] = tryParseJSON(value) ?? value
            } else {
                arguments[key] = value
            }

            searchRange = valueEnd.upperBound ..< text.endIndex
        }

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }

    /// Whether a bare `<tool_call>` body reads as a function identifier
    /// rather than as text the model happened to wrap in the envelope.
    /// Deliberately narrow: OpenAI-compatible tool names are
    /// `[A-Za-z0-9_.-]{1,64}`, and every name reaching this parser came from
    /// a schema that had to satisfy that.
    static func isFunctionNameShaped(_ name: String) -> Bool {
        guard (1 ... 64).contains(name.count) else { return false }
        return name.allSatisfy {
            $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "_" || $0 == "." || $0 == "-")
        }
    }
}
