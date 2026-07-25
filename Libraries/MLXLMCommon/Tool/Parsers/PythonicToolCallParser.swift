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

        guard let invocation = balancedFunctionInvocations(in: text).first,
            let arguments = parseArguments(
                invocation.arguments,
                funcName: invocation.name,
                tools: tools)
        else { return nil }
        let funcName = invocation.name
        guard acceptsToolCall(name: funcName, arguments: arguments, tools: tools) else {
            return nil
        }
        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }

    /// At end-of-sequence, extract every pythonic call in the buffer —
    /// Pythonic models legitimately emit multiple `name(args)` invocations
    /// inside one `[...]` block, and the default protocol `parseEOS` only
    /// surfaces the first. Parsing is balanced rather than regular-expression
    /// based so parentheses inside nested values or quoted strings cannot
    /// truncate a call or consume a following call.
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

        var results: [ToolCall] = []
        for invocation in balancedFunctionInvocations(in: text) {
            guard
                let arguments = parseArguments(
                    invocation.arguments,
                    funcName: invocation.name,
                    tools: tools),
                acceptsToolCall(
                    name: invocation.name,
                    arguments: arguments,
                    tools: tools)
            else {
                continue
            }
            results.append(
                ToolCall(function: .init(name: invocation.name, arguments: arguments)))
        }
        return results
    }

    private struct FunctionInvocation {
        let name: String
        let arguments: String
    }

    /// Find `name(...)` invocations while respecting nested parentheses and
    /// quoted strings. A truncated invocation is never returned. When the
    /// native bracket-list wrapper is present, it must also be complete.
    private func balancedFunctionInvocations(in input: String) -> [FunctionInvocation] {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        let text: String
        if trimmed.hasPrefix("[") {
            guard trimmed.hasSuffix("]") else { return [] }
            text = String(trimmed.dropFirst().dropLast())
        } else {
            text = trimmed
        }

        var invocations: [FunctionInvocation] = []
        var cursor = text.startIndex
        while cursor < text.endIndex {
            guard text[cursor].isLetter || text[cursor] == "_" else {
                cursor = text.index(after: cursor)
                continue
            }

            let nameStart = cursor
            cursor = text.index(after: cursor)
            while cursor < text.endIndex,
                text[cursor].isLetter || text[cursor].isNumber || text[cursor] == "_"
            {
                cursor = text.index(after: cursor)
            }
            let nameEnd = cursor
            while cursor < text.endIndex,
                text[cursor].unicodeScalars.allSatisfy({
                    CharacterSet.whitespacesAndNewlines.contains($0)
                })
            {
                cursor = text.index(after: cursor)
            }
            guard cursor < text.endIndex, text[cursor] == "(" else {
                continue
            }

            let openParen = cursor
            let argumentsStart = text.index(after: openParen)
            cursor = argumentsStart
            var depth = 1
            var quote: Character?
            var escaped = false
            var closeParen: String.Index?

            while cursor < text.endIndex {
                let character = text[cursor]
                if let activeQuote = quote {
                    if escaped {
                        escaped = false
                    } else if character == "\\" {
                        escaped = true
                    } else if character == activeQuote {
                        quote = nil
                    }
                } else {
                    switch character {
                    case "'", "\"":
                        quote = character
                    case "(":
                        depth += 1
                    case ")":
                        depth -= 1
                        if depth == 0 {
                            closeParen = cursor
                        }
                    default:
                        break
                    }
                }

                cursor = text.index(after: cursor)
                if closeParen != nil { break }
            }

            guard let closeParen, quote == nil, !escaped else {
                return invocations
            }
            invocations.append(
                FunctionInvocation(
                    name: String(text[nameStart ..< nameEnd]),
                    arguments: String(text[argumentsStart ..< closeParen])))
        }
        return invocations
    }

    /// Parse one Pythonic keyword-argument list.
    ///
    /// This is internal so parser families that explicitly recover the same
    /// native syntax can share one balanced implementation. It intentionally
    /// accepts only keyword arguments or the existing single-string
    /// positional form; malformed mixed fields are rejected as a whole.
    func parseArguments(
        _ argsString: String,
        funcName: String,
        tools: [[String: any Sendable]]?
    ) -> [String: any Sendable]? {
        guard let fields = splitTopLevelArguments(argsString) else {
            return nil
        }

        if fields.count == 1,
            splitKeywordArgument(fields[0]) == nil,
            let positionalString = parseSinglePositionalString(argsString),
            let firstRequiredParameter = requiredParameterNames(funcName: funcName, tools: tools)
                .first
        {
            return [
                firstRequiredParameter: convertParameterValue(
                    positionalString, paramName: firstRequiredParameter, funcName: funcName,
                    tools: tools)
            ]
        }

        var arguments: [String: any Sendable] = [:]
        for field in fields {
            guard let (key, rawValue) = splitKeywordArgument(field),
                arguments[key] == nil
            else {
                // Do not silently discard malformed or duplicate fields while
                // returning a superficially valid required-argument set.
                return nil
            }
            arguments[key] = decodeArgumentValue(
                rawValue,
                paramName: key,
                funcName: funcName,
                tools: tools)
        }
        return arguments
    }

    /// Split at commas only when outside strings and balanced containers.
    /// The previous regular expression stopped at the first comma inside a
    /// list or dictionary, corrupting nested `rows=[{...}]` arguments.
    private func splitTopLevelArguments(_ input: String) -> [String]? {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        var fields: [String] = []
        var start = input.startIndex
        var index = input.startIndex
        var quote: Character?
        var escaped = false
        var roundDepth = 0
        var squareDepth = 0
        var curlyDepth = 0

        while index < input.endIndex {
            let character = input[index]
            if let activeQuote = quote {
                if escaped {
                    escaped = false
                } else if character == "\\" {
                    escaped = true
                } else if character == activeQuote {
                    quote = nil
                }
            } else {
                switch character {
                case "'", "\"":
                    quote = character
                case "(":
                    roundDepth += 1
                case ")":
                    roundDepth -= 1
                case "[":
                    squareDepth += 1
                case "]":
                    squareDepth -= 1
                case "{":
                    curlyDepth += 1
                case "}":
                    curlyDepth -= 1
                case "," where roundDepth == 0 && squareDepth == 0 && curlyDepth == 0:
                    fields.append(String(input[start ..< index]))
                    start = input.index(after: index)
                default:
                    break
                }

                guard roundDepth >= 0, squareDepth >= 0, curlyDepth >= 0 else {
                    return nil
                }
            }
            index = input.index(after: index)
        }

        guard quote == nil,
            !escaped,
            roundDepth == 0,
            squareDepth == 0,
            curlyDepth == 0
        else {
            return nil
        }
        fields.append(String(input[start ..< input.endIndex]))
        return fields
    }

    private func splitKeywordArgument(_ field: String) -> (String, String)? {
        let trimmed = field.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        var index = trimmed.startIndex
        var quote: Character?
        var escaped = false
        var roundDepth = 0
        var squareDepth = 0
        var curlyDepth = 0

        while index < trimmed.endIndex {
            let character = trimmed[index]
            if let activeQuote = quote {
                if escaped {
                    escaped = false
                } else if character == "\\" {
                    escaped = true
                } else if character == activeQuote {
                    quote = nil
                }
            } else {
                switch character {
                case "'", "\"":
                    quote = character
                case "(":
                    roundDepth += 1
                case ")":
                    roundDepth -= 1
                case "[":
                    squareDepth += 1
                case "]":
                    squareDepth -= 1
                case "{":
                    curlyDepth += 1
                case "}":
                    curlyDepth -= 1
                case "=" where roundDepth == 0 && squareDepth == 0 && curlyDepth == 0:
                    let key = trimmed[..<index]
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    guard let first = key.first,
                        first.isLetter || first == "_",
                        key.allSatisfy({ $0.isLetter || $0.isNumber || $0 == "_" })
                    else {
                        return nil
                    }
                    let valueStart = trimmed.index(after: index)
                    let value = trimmed[valueStart...]
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !value.isEmpty else { return nil }
                    return (key, value)
                default:
                    break
                }
            }
            index = trimmed.index(after: index)
        }
        return nil
    }

    private func decodeArgumentValue(
        _ rawValue: String,
        paramName: String,
        funcName: String,
        tools: [[String: any Sendable]]?
    ) -> any Sendable {
        var value = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)

        if (value.hasPrefix("'") && value.hasSuffix("'"))
            || (value.hasPrefix("\"") && value.hasSuffix("\""))
        {
            value = String(value.dropFirst().dropLast())
            value = value.replacingOccurrences(of: "\\'", with: "'")
            value = value.replacingOccurrences(of: "\\\"", with: "\"")
            value = value.replacingOccurrences(of: "\\n", with: "\n")
            value = value.replacingOccurrences(of: "\\r", with: "\r")
            value = value.replacingOccurrences(of: "\\t", with: "\t")
            value = value.replacingOccurrences(of: "\\\\", with: "\\")
            return convertParameterValue(
                value, paramName: paramName, funcName: funcName, tools: tools)
        }

        let schemaType = getParameterType(
            funcName: funcName, paramName: paramName, tools: tools
        )?.lowercased()
        if schemaType != "string",
            schemaType == nil
                || schemaType == "array"
                || schemaType == "object"
                || schemaType?.hasPrefix("list") == true
                || schemaType?.hasPrefix("dict") == true,
            let container = parsePythonContainerLiteral(value)
        {
            return container
        }

        if schemaType != "string" {
            switch value {
            case "True": return true
            case "False": return false
            case "None": return NSNull()
            default: break
            }
        }
        return convertParameterValue(
            value, paramName: paramName, funcName: funcName, tools: tools)
    }

    private func parsePythonContainerLiteral(_ value: String) -> (any Sendable)? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard
            (trimmed.first == "[" && trimmed.last == "]")
                || (trimmed.first == "{" && trimmed.last == "}")
        else {
            return nil
        }
        guard let normalized = normalizePythonLiteralAsJSON(trimmed),
            let data = normalized.data(using: .utf8)
        else {
            return nil
        }
        return deserializeJSON(data)
    }

    /// Convert only the literal subset emitted by native Python-style tool
    /// templates: quoted strings, JSON containers, numbers, and the exact
    /// `True`/`False`/`None` scalars. This is not source evaluation.
    private func normalizePythonLiteralAsJSON(_ value: String) -> String? {
        let characters = Array(value)
        var output = ""
        var index = 0
        var quote: Character?
        var escaped = false

        while index < characters.count {
            let character = characters[index]
            if let activeQuote = quote {
                if escaped {
                    switch character {
                    case "'" where activeQuote == "'":
                        output.append("'")
                    case "'" where activeQuote == "\"":
                        output.append("'")
                    case "\"":
                        output.append("\\\"")
                    case "\\", "/", "b", "f", "n", "r", "t", "u":
                        output.append("\\")
                        output.append(character)
                    default:
                        return nil
                    }
                    escaped = false
                } else if character == "\\" {
                    escaped = true
                } else if character == activeQuote {
                    output.append("\"")
                    quote = nil
                } else if character == "\n" {
                    output.append("\\n")
                } else if character == "\r" {
                    output.append("\\r")
                } else if character == "\t" {
                    output.append("\\t")
                } else if character == "\"" {
                    output.append("\\\"")
                } else {
                    output.append(character)
                }
                index += 1
                continue
            }

            if character == "'" || character == "\"" {
                quote = character
                output.append("\"")
                index += 1
                continue
            }

            if character.isLetter || character == "_" {
                let start = index
                index += 1
                while index < characters.count,
                    characters[index].isLetter
                        || characters[index].isNumber
                        || characters[index] == "_"
                {
                    index += 1
                }
                let token = String(characters[start ..< index])
                switch token {
                case "True": output.append("true")
                case "False": output.append("false")
                case "None": output.append("null")
                case "true", "false", "null": output.append(token)
                default:
                    // Bare identifiers and executable expressions are outside
                    // the literal grammar and must not be guessed.
                    return nil
                }
                continue
            }

            output.append(character)
            index += 1
        }

        guard quote == nil, !escaped else { return nil }
        return output
    }

    private func parseSinglePositionalString(_ argsString: String) -> String? {
        let trimmed = argsString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count >= 2 else { return nil }

        let quote = trimmed.first
        guard quote == "'" || quote == "\"", trimmed.last == quote else { return nil }

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
