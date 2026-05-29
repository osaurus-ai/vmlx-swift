// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for Pythonic tool call format: [function_name(arg1='value1', arg2='value2')]
/// Used by LFM2.5 and similar models that output tool calls in Python function call syntax.
/// Reference: LiquidAI LFM2.5 chat template format
public struct PythonicToolCallParser: ToolCallParser, Sendable {
    public let startTag: String?
    public let endTag: String?
    public let supportsInlineJSONToolFallback = true

    public init(startTag: String? = nil, endTag: String? = nil) {
        self.startTag = startTag
        self.endTag = endTag
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        var text = content

        // Strip tags if present
        if let start = startTag, let startRange = text.range(of: start) {
            text = String(text[startRange.upperBound...])
        }
        if let end = endTag, let endRange = text.range(of: end) {
            text = String(text[..<endRange.lowerBound])
        }

        text = text.trimmingCharacters(in: .whitespacesAndNewlines)

        let funcName: String
        let argsString: String

        // Required brackets pattern (matches Python reference: r"\[(\w+)\((.*?)\)\]")
        // The required \] forces .*? to backtrack past nested ) inside argument values.
        let bracketPattern = #"\[(\w+)\((.*?)\)\]"#
        if let regex = try? NSRegularExpression(
            pattern: bracketPattern, options: [.dotMatchesLineSeparators]),
            let match = regex.firstMatch(
                in: text, options: [], range: NSRange(text.startIndex..., in: text)),
            let nameRange = Range(match.range(at: 1), in: text),
            let argsRange = Range(match.range(at: 2), in: text)
        {
            funcName = String(text[nameRange])
            argsString = String(text[argsRange])
        } else {
            // Fallback for without-brackets case: use string indices to find the
            // outermost parentheses, avoiding the greedy/non-greedy regex pitfall.
            guard let openParen = text.firstIndex(of: "("),
                let closeParen = text.lastIndex(of: ")")
            else { return nil }

            let name = text[text.startIndex ..< openParen]
            guard !name.isEmpty, name.allSatisfy({ $0.isLetter || $0.isNumber || $0 == "_" })
            else { return nil }

            funcName = String(name)
            argsString = String(text[text.index(after: openParen) ..< closeParen])
        }

        let arguments = parseArguments(argsString, funcName: funcName, tools: tools)
        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }

    /// At end-of-sequence, extract every pythonic call in the buffer —
    /// Pythonic models legitimately emit multiple `name(args)` invocations
    /// inside one `[...]` block, and the default protocol `parseEOS` only
    /// surfaces the first. Byte-compatible with upstream
    /// ml-explore/mlx-swift-lm so `LFM2` streams behave identically.
    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        if let startTag {
            return
                toolCallBuffer
                .components(separatedBy: startTag)
                .filter { !$0.isEmpty }
                .flatMap { parseMultiple(content: $0, tools: tools) }
        } else {
            return parseMultiple(content: toolCallBuffer, tools: tools)
        }
    }

    private func parseMultiple(content: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        var text = content

        if let end = endTag, let endRange = text.range(of: end) {
            text = String(text[..<endRange.lowerBound])
        }

        text = text.trimmingCharacters(in: .whitespacesAndNewlines)

        // Match every `name(args)` — NSRegularExpression (not Swift regex
        // literal) keeps us compatible with older Swift toolchains that
        // don't parse `#/(?s)(\w+)\((.*?)\)/#` at compile time. The `(?s)`
        // dotall flag is expressed via `.dotMatchesLineSeparators`.
        let pattern = #"(\w+)\(([^)]*(?:\)[^)]*)*?)\)"#
        guard let regex = try? NSRegularExpression(
            pattern: pattern, options: [.dotMatchesLineSeparators])
        else { return [] }
        let matches = regex.matches(
            in: text, options: [], range: NSRange(text.startIndex..., in: text))

        var results: [ToolCall] = []
        for match in matches {
            guard let nameRange = Range(match.range(at: 1), in: text),
                let argsRange = Range(match.range(at: 2), in: text)
            else { continue }
            let funcName = String(text[nameRange])
            let argsString = String(text[argsRange])
            let arguments = parseArguments(argsString, funcName: funcName, tools: tools)
            results.append(ToolCall(function: .init(name: funcName, arguments: arguments)))
        }
        return results
    }

    /// Parse Pythonic keyword arguments: arg1='value1', arg2="value2", arg3=123
    private func parseArguments(
        _ argsString: String,
        funcName: String,
        tools: [[String: any Sendable]]?
    ) -> [String: any Sendable] {
        var arguments: [String: any Sendable] = [:]

        // Pattern for key=value pairs, handling quoted strings with possible commas inside
        // This handles: key='value', key="value", key=123, key=True, key=None
        let argPattern = #"(\w+)\s*=\s*('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|[^,\)]+)"#

        guard let regex = try? NSRegularExpression(pattern: argPattern, options: []) else {
            return arguments
        }

        let matches = regex.matches(
            in: argsString, options: [], range: NSRange(argsString.startIndex..., in: argsString))

        for match in matches {
            guard let keyRange = Range(match.range(at: 1), in: argsString),
                let valueRange = Range(match.range(at: 2), in: argsString)
            else { continue }

            let key = String(argsString[keyRange])
            var value = String(argsString[valueRange]).trimmingCharacters(in: .whitespaces)

            // Remove surrounding quotes if present
            if (value.hasPrefix("'") && value.hasSuffix("'"))
                || (value.hasPrefix("\"") && value.hasSuffix("\""))
            {
                value = String(value.dropFirst().dropLast())
                // Unescape escaped quotes
                value = value.replacingOccurrences(of: "\\'", with: "'")
                value = value.replacingOccurrences(of: "\\\"", with: "\"")
                value = value.replacingOccurrences(of: "\\n", with: "\n")
                value = value.replacingOccurrences(of: "\\t", with: "\t")
                value = value.replacingOccurrences(of: "\\\\", with: "\\")
            }

            // Convert value based on schema type if available
            arguments[key] = convertParameterValue(
                value, paramName: key, funcName: funcName, tools: tools)
        }

        if arguments.isEmpty,
            let positionalString = parseSinglePositionalString(argsString),
            let firstRequiredParameter = requiredParameterNames(funcName: funcName, tools: tools).first
        {
            arguments[firstRequiredParameter] = convertParameterValue(
                positionalString, paramName: firstRequiredParameter, funcName: funcName, tools: tools)
        }

        return arguments
    }

    private func parseSinglePositionalString(_ argsString: String) -> String? {
        let trimmed = argsString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count >= 2 else { return nil }

        let quote = trimmed.first
        guard (quote == "'" || quote == "\""), trimmed.last == quote else { return nil }

        var value = String(trimmed.dropFirst().dropLast())
        value = value.replacingOccurrences(of: "\\'", with: "'")
        value = value.replacingOccurrences(of: "\\\"", with: "\"")
        value = value.replacingOccurrences(of: "\\n", with: "\n")
        value = value.replacingOccurrences(of: "\\t", with: "\t")
        value = value.replacingOccurrences(of: "\\\\", with: "\\")
        return value
    }

    private func requiredParameterNames(
        funcName: String,
        tools: [[String: any Sendable]]?
    ) -> [String] {
        guard let tools else { return [] }
        for tool in tools {
            guard let function = tool["function"] as? [String: any Sendable],
                function["name"] as? String == funcName,
                let parameters = function["parameters"] as? [String: any Sendable],
                let required = parameters["required"] as? [String]
            else { continue }
            return required
        }
        return []
    }
}
