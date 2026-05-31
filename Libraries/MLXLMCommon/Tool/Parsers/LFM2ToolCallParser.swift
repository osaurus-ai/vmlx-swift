// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// LFM2/LFM2.5 tool-call parser.
///
/// The native Liquid template is Pythonic:
///
///     <|tool_call_start|>[line_count(text="red\nblue")]<|tool_call_end|>
///
/// Live LFM2.5 JANG rows can also emit malformed-but-explicit function tags
/// when `tool_choice: required` is active. Keep that recovery narrow: only
/// accept `<function...>` protocol-looking text, require schema-compatible
/// arguments, and never treat plain code fences or copied prose as tool calls.
public struct LFM2ToolCallParser: ToolCallParser, Sendable {
    private let native = PythonicToolCallParser(
        startTag: "<|tool_call_start|>",
        endTag: "<|tool_call_end|>")

    public let startTag: String? = "<|tool_call_start|>"
    public let endTag: String? = "<|tool_call_end|>"
    public let supportsInlineJSONToolFallback = true

    public var startTagAliases: [String] { native.startTagAliases }
    public var endTagAliases: [String] { native.endTagAliases }

    /// Buffer observed `<function...>` LFM attempts until EOS. They do not
    /// have a reliable close tag shape, and parsing only at EOS prevents
    /// half-markers from leaking to the visible answer.
    public let startTagPrefixes: [String] = ["<function"]
    public let endTagPrefixes: [String] = []

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        if let call = native.parse(content: content, tools: tools) {
            return call
        }
        return parseObservedFunctionishCall(content, tools: tools)
    }

    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        let nativeCalls = native.parseEOS(toolCallBuffer, tools: tools)
        if !nativeCalls.isEmpty {
            return nativeCalls
        }
        guard let call = parseObservedFunctionishCall(toolCallBuffer, tools: tools) else {
            return []
        }
        return [call]
    }

    public func isValidPartialContent(_ toolCallBuffer: String) -> Bool {
        let trimmed = toolCallBuffer.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return true }
        return "<function".hasPrefix(trimmed) || trimmed.hasPrefix("<function")
            || trimmed.hasPrefix(startTag ?? "")
    }

    private func parseObservedFunctionishCall(
        _ content: String,
        tools: [[String: any Sendable]]?
    ) -> ToolCall? {
        let text = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard text.hasPrefix("<function") || text.contains("<function") else {
            return nil
        }

        let extractedName = observedFunctionName(in: text, tools: tools)
        let spec: ToolSpecInfo
        if let name = extractedName, let namedSpec = toolSpec(named: name, tools: tools) {
            spec = namedSpec
        } else if let single = singleToolSpec(tools: tools) {
            spec = single
        } else {
            return nil
        }

        var args = observedArguments(in: text, spec: spec)
        if args.isEmpty,
            let required = spec.required.first,
            spec.required.count == 1,
            let scalar = observedSingleRequiredScalar(in: text, toolName: spec.name)
        {
            args[required] = convertParameterValue(
                scalar, paramName: required, funcName: spec.name, tools: tools)
        }

        guard spec.required.allSatisfy({ args[$0] != nil }) else {
            return nil
        }
        if !spec.properties.isEmpty {
            guard args.keys.allSatisfy({ spec.properties.contains($0) }) else {
                return nil
            }
        }
        return ToolCall(function: .init(name: spec.name, arguments: args))
    }

    private func observedFunctionName(
        in text: String,
        tools: [[String: any Sendable]]?
    ) -> String? {
        if let name = text.between("<functionname>", and: "</functionname>")?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            isIdentifier(name)
        {
            return name
        }

        if let name = firstRegexCapture(
            #"<functionname>\s*([A-Za-z_][A-Za-z0-9_]*)\s*</functionname>"#,
            in: text)
        {
            return name
        }

        if let suffix = firstRegexCapture(#"<function([A-Za-z_][A-Za-z0-9_]*)>"#, in: text),
            toolSpec(named: suffix, tools: tools) != nil
        {
            return suffix
        }

        for name in toolNames(tools: tools).sorted(by: { $0.count > $1.count }) {
            if text.contains(name) {
                return name
            }
        }
        return nil
    }

    private func observedArguments(
        in text: String,
        spec: ToolSpecInfo
    ) -> [String: any Sendable] {
        var args: [String: any Sendable] = [:]
        for (key, value) in scanObservedArgTags(in: text) {
            args[key] = convertParameterValue(
                decodeObservedScalar(value),
                paramName: key,
                funcName: spec.name,
                tools: nil)
        }

        for match in regexCaptures(
            #"<arg[^>]*\bname\s*=\s*["']([A-Za-z_][A-Za-z0-9_]*)["'][^>]*>(.*?)</arg[^>]*>"#,
            in: text)
        {
            guard match.count == 2 else { continue }
            let key = match[0]
            args[key] = convertParameterValue(
                decodeObservedScalar(match[1]),
                paramName: key,
                funcName: spec.name,
                tools: nil)
        }

        if args.isEmpty,
            let argOpen = text.range(of: "<arg"),
            let tagEnd = text[argOpen.upperBound...].firstIndex(of: ">"),
            let key = attributeValue(named: "name", in: String(text[argOpen.lowerBound...tagEnd])),
            isIdentifier(key),
            let close = text[text.index(after: tagEnd)...].range(of: "</arg")
        {
            let valueStart = text.index(after: tagEnd)
            args[key] = convertParameterValue(
                decodeObservedScalar(String(text[valueStart..<close.lowerBound])),
                paramName: key,
                funcName: spec.name,
                tools: nil)
        }

        for match in regexCaptures(
            #"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*('(?:[^'\\]|\\.|[\n\r])*'|"(?:[^"\\]|\\.|[\n\r])*")"#,
            in: text)
        {
            guard match.count == 2 else { continue }
            let key = match[0]
            guard spec.properties.isEmpty || spec.properties.contains(key) else { continue }
            var value = match[1]
            if value.count >= 2 {
                value.removeFirst()
                value.removeLast()
            }
            args[key] = convertParameterValue(
                decodeObservedScalar(value),
                paramName: key,
                funcName: spec.name,
                tools: nil)
        }
        return args
    }

    private func scanObservedArgTags(in text: String) -> [(String, String)] {
        var results: [(String, String)] = []
        var cursor = text.startIndex
        while let open = text[cursor...].range(of: "<arg") {
            guard let tagEnd = text[open.upperBound...].firstIndex(of: ">") else { break }
            let tag = String(text[open.lowerBound...tagEnd])
            guard let key = attributeValue(named: "name", in: tag), isIdentifier(key) else {
                cursor = text.index(after: tagEnd)
                continue
            }

            let valueStart = text.index(after: tagEnd)
            guard let closeStart = text[valueStart...].range(of: "</arg") else { break }
            results.append((key, String(text[valueStart..<closeStart.lowerBound])))
            cursor = closeStart.upperBound
        }
        return results
    }

    private func attributeValue(named name: String, in tag: String) -> String? {
        for quote in ["\"", "'"] {
            let prefix = "\(name)=\(quote)"
            guard let start = tag.range(of: prefix) else { continue }
            let valueStart = start.upperBound
            guard let valueEnd = tag[valueStart...].firstIndex(of: Character(quote)) else {
                continue
            }
            return String(tag[valueStart..<valueEnd])
        }
        return nil
    }

    private func observedSingleRequiredScalar(in text: String, toolName: String) -> String? {
        guard let open = text.range(of: "<function\(toolName)>") else {
            return nil
        }
        let afterOpen = String(text[open.upperBound...])
        guard let close = afterOpen.range(of: "</function") else {
            return nil
        }
        let value = String(afterOpen[..<close.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty, value != toolName else { return nil }
        return decodeObservedScalar(value)
    }

    private func decodeObservedScalar(_ value: String) -> String {
        value
            .replacingOccurrences(of: "\\n", with: "\n")
            .replacingOccurrences(of: "\\r", with: "\r")
            .replacingOccurrences(of: "\\t", with: "\t")
            .replacingOccurrences(of: "\\'", with: "'")
            .replacingOccurrences(of: "\\\"", with: "\"")
            .replacingOccurrences(of: "\\\\", with: "\\")
    }

    private func firstRegexCapture(_ pattern: String, in text: String) -> String? {
        regexCaptures(pattern, in: text).first?.first
    }

    private func isIdentifier(_ value: String) -> Bool {
        guard let first = value.first,
            first.isLetter || first == "_"
        else { return false }
        return value.allSatisfy { $0.isLetter || $0.isNumber || $0 == "_" }
    }

    private func regexCaptures(_ pattern: String, in text: String) -> [[String]] {
        guard let regex = try? NSRegularExpression(
            pattern: pattern,
            options: [.dotMatchesLineSeparators])
        else { return [] }
        return regex.matches(
            in: text,
            options: [],
            range: NSRange(text.startIndex..., in: text)
        ).map { match in
            (1..<match.numberOfRanges).compactMap { index in
                guard let range = Range(match.range(at: index), in: text) else { return nil }
                return String(text[range])
            }
        }
    }

    private struct ToolSpecInfo {
        let name: String
        let required: [String]
        let properties: Set<String>
    }

    private func singleToolSpec(tools: [[String: any Sendable]]?) -> ToolSpecInfo? {
        guard let tools, tools.count == 1,
            let function = tools[0]["function"] as? [String: any Sendable],
            let name = function["name"] as? String
        else { return nil }
        return spec(from: function, name: name)
    }

    private func toolSpec(
        named name: String,
        tools: [[String: any Sendable]]?
    ) -> ToolSpecInfo? {
        guard let tools else { return nil }
        for tool in tools {
            let function = (tool["function"] as? [String: any Sendable]) ?? tool
            guard function["name"] as? String == name else { continue }
            return spec(from: function, name: name)
        }
        return nil
    }

    private func spec(
        from function: [String: any Sendable],
        name: String
    ) -> ToolSpecInfo {
        let parameters = function["parameters"] as? [String: any Sendable]
        let properties = parameters?["properties"] as? [String: any Sendable]
        return ToolSpecInfo(
            name: name,
            required: parameters?["required"] as? [String] ?? [],
            properties: properties.map { Set($0.keys) } ?? [])
    }

    private func toolNames(tools: [[String: any Sendable]]?) -> [String] {
        guard let tools else { return [] }
        return tools.compactMap { tool in
            let function = (tool["function"] as? [String: any Sendable]) ?? tool
            return function["name"] as? String
        }
    }
}

private extension String {
    func between(_ start: String, and end: String) -> String? {
        guard let startRange = range(of: start) else { return nil }
        let remainder = self[startRange.upperBound...]
        guard let endRange = remainder.range(of: end) else { return nil }
        return String(remainder[..<endRange.lowerBound])
    }
}
