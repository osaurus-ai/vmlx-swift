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

        if let jsonToolCall = parseToolKeyedJSONEnvelope(text, tools: tools) {
            return jsonToolCall
        }
        if let jsonToolCall = parseOpenAIToolCallJSONEnvelope(text, tools: tools) {
            return jsonToolCall
        }
        if let jsonToolCall = parseNamedArgumentsJSONEnvelope(text, tools: tools) {
            return jsonToolCall
        }
        if let jsonToolCall = parseSingleToolArgumentsJSON(text, tools: tools) {
            return jsonToolCall
        }
        if let jsonToolCall = parseBracketedNameAndArgumentsArray(text, tools: tools) {
            return jsonToolCall
        }

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
        guard acceptsToolCall(name: funcName, arguments: arguments, tools: tools) else {
            return nil
        }
        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }

    /// At end-of-sequence, extract every pythonic call in the buffer —
    /// Pythonic models legitimately emit multiple `name(args)` invocations
    /// inside one `[...]` block, and the default protocol `parseEOS` only
    /// surfaces the first. Byte-compatible with upstream
    /// ml-explore/mlx-swift-lm so `LFM2` streams behave identically.
    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        if let startTag {
            let calls =
                toolCallBuffer
                .components(separatedBy: startTag)
                .filter { !$0.isEmpty }
                .flatMap { parseMultiple(content: $0, tools: tools) }
            if !calls.isEmpty {
                return calls
            }
        } else {
            let calls = parseMultiple(content: toolCallBuffer, tools: tools)
            if !calls.isEmpty {
                return calls
            }
        }

        if let toolCall = parse(content: toolCallBuffer, tools: tools) {
            return [toolCall]
        }
        return []
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
            guard acceptsToolCall(name: funcName, arguments: arguments, tools: tools) else {
                continue
            }
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

    private func acceptsToolCall(
        name: String,
        arguments: [String: any Sendable],
        tools: [[String: any Sendable]]?
    ) -> Bool {
        guard let tools, !tools.isEmpty else { return true }
        guard toolNames(tools: tools).contains(name) else { return false }

        let spec = toolSpec(named: name, tools: tools)
        if let required = spec.required, !required.isEmpty {
            guard required.allSatisfy({ arguments[$0] != nil }) else { return false }
        }
        if let properties = spec.properties, !properties.isEmpty {
            guard arguments.keys.allSatisfy({ properties.contains($0) }) else { return false }
        }
        return true
    }

    private func parseToolKeyedJSONEnvelope(
        _ text: String,
        tools: [[String: any Sendable]]?
    ) -> ToolCall? {
        guard text.hasPrefix("{") else { return nil }
        guard let object = parseJSONObjectWithOptionalEOFBrace(text) else { return nil }
        let registeredNames = Set(toolNames(tools: tools))
        guard !registeredNames.isEmpty else { return nil }

        for name in registeredNames {
            guard let rawArguments = object[name] else { continue }
            if let args = firstArgumentObject(from: rawArguments) {
                return ToolCall(function: .init(name: name, arguments: args.mapValues(asSendable)))
            }
        }
        return nil
    }

    private func parseOpenAIToolCallJSONEnvelope(
        _ text: String,
        tools: [[String: any Sendable]]?
    ) -> ToolCall? {
        guard text.hasPrefix("{") else { return nil }
        guard let object = parseJSONObjectWithOptionalEOFBrace(text) else { return nil }
        guard object.count == 1 else { return nil }

        let rawCall = object["tool_call"] ?? object["toolCall"]
        guard let callObject = rawCall as? [String: Any],
            let name = callObject["name"] as? String,
            toolNames(tools: tools).contains(name),
            let rawArguments = callObject["arguments"],
            let args = firstArgumentObject(from: rawArguments)
        else { return nil }

        let spec = toolSpec(named: name, tools: tools)
        if let required = spec.required, !required.isEmpty {
            guard required.allSatisfy({ args[$0] != nil }) else { return nil }
        }
        if let properties = spec.properties, !properties.isEmpty {
            guard args.keys.allSatisfy({ properties.contains($0) }) else { return nil }
        }
        if let type = callObject["type"] as? String,
            type != "function" && type != "tool_call"
        {
            return nil
        }
        return ToolCall(function: .init(name: name, arguments: args.mapValues(asSendable)))
    }

    private func parseSingleToolArgumentsJSON(
        _ text: String,
        tools: [[String: any Sendable]]?
    ) -> ToolCall? {
        guard text.hasPrefix("{") else { return nil }
        guard let object = parseJSONObjectWithOptionalEOFBrace(text) else { return nil }
        guard let tool = singleToolSpec(tools: tools),
            let name = tool.name,
            let required = tool.required,
            !required.isEmpty
        else { return nil }

        if object.count == 1,
            let rawArguments = object["arguments"],
            let args = firstArgumentObject(from: rawArguments)
        {
            guard required.allSatisfy({ args[$0] != nil }) else {
                return nil
            }
            if let properties = tool.properties, !properties.isEmpty {
                guard args.keys.allSatisfy({ properties.contains($0) }) else {
                    return nil
                }
            }
            return ToolCall(function: .init(name: name, arguments: args.mapValues(asSendable)))
        }

        let registeredNames = Set(toolNames(tools: tools))
        guard object.keys.allSatisfy({ !registeredNames.contains($0) }) else {
            return nil
        }
        guard required.allSatisfy({ object[$0] != nil }) else {
            return nil
        }
        if let properties = tool.properties, !properties.isEmpty {
            guard object.keys.allSatisfy({ properties.contains($0) }) else {
                return nil
            }
        }
        return ToolCall(function: .init(name: name, arguments: object.mapValues(asSendable)))
    }

    private func parseNamedArgumentsJSONEnvelope(
        _ text: String,
        tools: [[String: any Sendable]]?
    ) -> ToolCall? {
        guard text.hasPrefix("{") else { return nil }
        guard let object = parseJSONObjectWithOptionalEOFBrace(text) else { return nil }
        guard object.count == 2 || object.count == 3 else { return nil }
        guard let name = object["name"] as? String,
            toolNames(tools: tools).contains(name),
            let rawArguments = object["arguments"],
            let args = firstArgumentObject(from: rawArguments)
        else { return nil }

        let spec = toolSpec(named: name, tools: tools)
        if let required = spec.required, !required.isEmpty {
            guard required.allSatisfy({ args[$0] != nil }) else { return nil }
        }
        if let properties = spec.properties, !properties.isEmpty {
            guard args.keys.allSatisfy({ properties.contains($0) }) else { return nil }
        }
        if let type = object["type"] as? String,
            type != "function" && type != "tool_call"
        {
            return nil
        }
        return ToolCall(function: .init(name: name, arguments: args.mapValues(asSendable)))
    }

    private func parseBracketedNameAndArgumentsArray(
        _ text: String,
        tools: [[String: any Sendable]]?
    ) -> ToolCall? {
        guard text.hasPrefix("[") else { return nil }
        guard let data = text.data(using: .utf8),
            let array = try? JSONSerialization.jsonObject(with: data) as? [Any],
            array.count == 2,
            let name = array[0] as? String,
            toolNames(tools: tools).contains(name),
            let args = array[1] as? [String: Any]
        else { return nil }

        let spec = toolSpec(named: name, tools: tools)
        if let required = spec.required, !required.isEmpty {
            guard required.allSatisfy({ args[$0] != nil }) else { return nil }
        }
        if let properties = spec.properties, !properties.isEmpty {
            guard args.keys.allSatisfy({ properties.contains($0) }) else { return nil }
        }
        return ToolCall(function: .init(name: name, arguments: args.mapValues(asSendable)))
    }

    private func parseJSONObjectWithOptionalEOFBrace(_ text: String) -> [String: Any]? {
        if let object = parseJSONObject(text) {
            return object
        }

        let missingCloseBraceCount = text.reduce(0) { depth, ch in
            if ch == "{" { return depth + 1 }
            if ch == "}" { return depth - 1 }
            return depth
        }
        guard missingCloseBraceCount > 0, missingCloseBraceCount <= 2 else {
            return nil
        }
        return parseJSONObject(text + String(repeating: "}", count: missingCloseBraceCount))
    }

    private func parseJSONObject(_ text: String) -> [String: Any]? {
        guard let data = text.data(using: .utf8) else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    private func firstArgumentObject(from rawArguments: Any) -> [String: Any]? {
        if let object = rawArguments as? [String: Any] {
            return object
        }
        if let array = rawArguments as? [Any] {
            return array.compactMap { $0 as? [String: Any] }.first
        }
        return nil
    }

    private func toolNames(tools: [[String: any Sendable]]?) -> [String] {
        guard let tools else { return [] }
        return tools.compactMap { tool in
            guard let function = tool["function"] as? [String: any Sendable] else { return nil }
            return function["name"] as? String
        }
    }

    private func singleToolSpec(
        tools: [[String: any Sendable]]?
    ) -> (name: String?, required: [String]?, properties: Set<String>?)? {
        guard let tools, tools.count == 1,
            let function = tools[0]["function"] as? [String: any Sendable]
        else { return nil }
        let parameters = function["parameters"] as? [String: any Sendable]
        let properties = parameters?["properties"] as? [String: any Sendable]
        return (
            function["name"] as? String,
            parameters?["required"] as? [String],
            properties.map { Set($0.keys) }
        )
    }

    private func toolSpec(
        named name: String,
        tools: [[String: any Sendable]]?
    ) -> (required: [String]?, properties: Set<String>?) {
        guard let tools else { return (nil, nil) }
        for tool in tools {
            guard let function = tool["function"] as? [String: any Sendable],
                function["name"] as? String == name
            else { continue }
            let parameters = function["parameters"] as? [String: any Sendable]
            let properties = parameters?["properties"] as? [String: any Sendable]
            return (
                parameters?["required"] as? [String],
                properties.map { Set($0.keys) }
            )
        }
        return (nil, nil)
    }
}
