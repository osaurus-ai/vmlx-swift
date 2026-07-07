// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for XML function format: <function=name><parameter=key>value</parameter></function>
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/qwen3_coder.py
public struct XMLFunctionParser: ToolCallParser, Sendable {
    public let startTag: String?
    public let endTag: String?
    public let decodesHTMLLineBreaks: Bool
    public let unwrapJSONQuotedStringParameters: Bool

    public init(
        startTag: String,
        endTag: String,
        decodesHTMLLineBreaks: Bool = false,
        unwrapJSONQuotedStringParameters: Bool = false
    ) {
        self.startTag = startTag
        self.endTag = endTag
        self.decodesHTMLLineBreaks = decodesHTMLLineBreaks
        self.unwrapJSONQuotedStringParameters = unwrapJSONQuotedStringParameters
    }

    /// The XML-function transport's closers are protocol control markers even
    /// when orphaned (no matching opener): live ZAYA rows emit stray
    /// `</parameter></function></zyphra_tool_call>` runs mid-conversation
    /// (MODEL_ISSUES_TRIAGE Issue 3 — the wrapper tags are dedicated special
    /// tokens for that family). Register the wrapper closer plus the body
    /// closers so the streaming processor strips an orphan run instead of
    /// surfacing it as visible assistant text.
    public var orphanStripTags: [String] {
        endTagAliases + ["</function>", "</parameter>"]
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        // Some ZAYA rows emit a fully *nested* XML body instead of the
        // attribute form this parser was written for:
        //
        //   <function>file_write
        //   <parameter>
        //   <name>path</name>
        //   <value>out.txt</value>
        //   </parameter>
        //
        // i.e. `<function>name` (no `=`) and `<parameter><name>k</name>
        // <value>v</value></parameter>` instead of `<function=name>` /
        // `<parameter=k>v</parameter>`. The attribute scan below finds no
        // `<parameter=` and drops every argument (observed: file_write parses
        // but `path` is reported missing). Handle the nested form first — gated
        // on the distinctive `<name>`-inside-`<parameter>` marker so the
        // attribute path is left completely untouched (no regression). See
        // MODEL_ISSUES_TRIAGE / ZAYA tool-arg extraction.
        if let nested = parseNestedParameterForm(content, tools: tools) {
            return nested
        }
        // Pattern: <function=(content)</function> — [\s\S] matches newlines
        guard
            let funcMatch = content.range(
                of: #"<function=([\s\S]*?)</function>"#, options: .regularExpression)
        else {
            // The `qwen` capability stamp covers two wire formats: Qwen3-Coder
            // emits the `<function=…><parameter=…>` XML handled above, but
            // Qwen3 (non-coder) and deepseek_v3-arch bundles such as
            // Kanana-2-30B-A3B emit a JSON object inside the `<tool_call>` tags
            // (`<tool_call>\n{"name": …, "arguments": …}\n</tool_call>`) per their
            // chat template. The XML pattern never matches that, so the call
            // would be silently dropped. Fall back to JSON decoding — only when
            // no `<function=>` block is present and the body is a JSON object —
            // so the same stamp drives both forms with no regression to the XML
            // path (it is still tried first) and no risk to non-tool prose.
            return parseJSONToolCallFallback(content)
        }

        let funcContent = String(content[funcMatch])

        // Extract function name. Most models emit `<function=name>`, but
        // live Zyphra/Gemma-family rows can put the nested parameter tag on
        // the next line before closing the function opener:
        //
        //   <function=line_count
        //   <parameter=text
        //   >...
        //
        // Treat that as the same XML-function transport instead of leaking a
        // protocol block as visible assistant text.
        guard let nameStart = funcContent.range(of: "<function=") else {
            return nil
        }

        let firstParameter = funcContent.range(
            of: "<parameter=", range: nameStart.upperBound ..< funcContent.endIndex)
        let firstHeaderEnd = funcContent.range(
            of: ">", range: nameStart.upperBound ..< funcContent.endIndex)

        let nameEnd: String.Index
        let paramSectionStart: String.Index
        if let firstParameter,
            let firstHeaderEnd,
            firstParameter.lowerBound < firstHeaderEnd.lowerBound
        {
            nameEnd = firstParameter.lowerBound
            paramSectionStart = firstParameter.lowerBound
        } else if let firstHeaderEnd {
            nameEnd = firstHeaderEnd.lowerBound
            paramSectionStart = firstHeaderEnd.upperBound
        } else if let firstParameter {
            nameEnd = firstParameter.lowerBound
            paramSectionStart = firstParameter.lowerBound
        } else {
            return nil
        }

        let funcName = String(funcContent[nameStart.upperBound ..< nameEnd])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !funcName.isEmpty else { return nil }
        let paramSection = String(funcContent[paramSectionStart...])

        var arguments: [String: any Sendable] = [:]

        // Find all parameter tags
        var searchRange = paramSection.startIndex ..< paramSection.endIndex
        while let paramStart = paramSection.range(of: "<parameter=", range: searchRange) {
            // Find the parameter name (between = and >)
            guard
                let nameEnd = paramSection.range(
                    of: ">", range: paramStart.upperBound ..< paramSection.endIndex)
            else { break }

            let paramName = String(paramSection[paramStart.upperBound ..< nameEnd.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !paramName.isEmpty else {
                searchRange = nameEnd.upperBound ..< paramSection.endIndex
                continue
            }

            // Find the closing </parameter> tag
            guard
                let paramEnd = paramSection.range(
                    of: "</parameter>", range: nameEnd.upperBound ..< paramSection.endIndex)
            else { break }

            var paramValue = String(paramSection[nameEnd.upperBound ..< paramEnd.lowerBound])

            // Trim leading/trailing newlines (matching Python behavior)
            paramValue = trimBoundaryNewlines(paramValue)

            if decodesHTMLLineBreaks,
               isStringType(funcName: funcName, argName: paramName, tools: tools) {
                paramValue = decodeHTMLLineBreaks(paramValue)
            }
            if unwrapJSONQuotedStringParameters,
               isStringType(funcName: funcName, argName: paramName, tools: tools),
               let unwrapped = decodeQuotedStringParameter(paramValue) {
                paramValue = trimBoundaryNewlines(unwrapped)
            }

            // Convert value based on schema type
            arguments[paramName] = convertParameterValue(
                paramValue, paramName: paramName, funcName: funcName, tools: tools)

            searchRange = paramEnd.upperBound ..< paramSection.endIndex
        }

        if let invalidArguments = schemaValidationFailure(
            toolName: funcName,
            arguments: arguments,
            tools: tools)
        {
            return ToolCall(function: .init(name: funcName, arguments: invalidArguments))
        }

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }

    /// Parse the *nested* ZAYA XML body:
    /// `<function>name … <parameter><name>key</name><value>val</value></parameter> …`.
    /// Returns nil — so the caller falls through to the attribute-style scan —
    /// unless the distinctive nested `<name>…</name>` / `<value>…</value>`
    /// parameter markers are present, so attribute-style calls are never
    /// rerouted here. Reuses the same value post-processing / schema validation
    /// as the attribute path so both wire forms behave identically downstream.
    private func parseNestedParameterForm(
        _ content: String,
        tools: [[String: any Sendable]]?
    ) -> ToolCall? {
        // Gate on the nested markers so the attribute path stays untouched.
        guard content.range(of: "<name>") != nil,
            content.range(of: "<value>") != nil,
            let funcOpen = content.range(of: "<function")
        else { return nil }

        // Function name: `<function>name` (nested) or `<function=name>`
        // (attribute) — read from just after the opener to the first
        // newline / `<` / `>`.
        var cursor = funcOpen.upperBound
        if cursor < content.endIndex, content[cursor] == "=" || content[cursor] == ">" {
            cursor = content.index(after: cursor)
        }
        let nameStop =
            content[cursor...].firstIndex { $0 == "\n" || $0 == "<" || $0 == ">" }
            ?? content.endIndex
        let funcName = String(content[cursor ..< nameStop])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !funcName.isEmpty else { return nil }

        // Pair each `<name>key</name>` with the following `<value>val</value>`.
        var arguments: [String: any Sendable] = [:]
        var search = content.startIndex ..< content.endIndex
        while let nameOpen = content.range(of: "<name>", range: search),
            let nameClose = content.range(
                of: "</name>", range: nameOpen.upperBound ..< content.endIndex)
        {
            let key = String(content[nameOpen.upperBound ..< nameClose.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            search = nameClose.upperBound ..< content.endIndex
            guard !key.isEmpty,
                let valueOpen = content.range(of: "<value>", range: search),
                let valueClose = content.range(
                    of: "</value>", range: valueOpen.upperBound ..< content.endIndex)
            else { continue }

            var value = String(content[valueOpen.upperBound ..< valueClose.lowerBound])
            value = trimBoundaryNewlines(value)
            if decodesHTMLLineBreaks,
                isStringType(funcName: funcName, argName: key, tools: tools) {
                value = decodeHTMLLineBreaks(value)
            }
            if unwrapJSONQuotedStringParameters,
                isStringType(funcName: funcName, argName: key, tools: tools),
                let unwrapped = decodeQuotedStringParameter(value) {
                value = trimBoundaryNewlines(unwrapped)
            }
            arguments[key] = convertParameterValue(
                value, paramName: key, funcName: funcName, tools: tools)
            search = valueClose.upperBound ..< content.endIndex
        }

        // No nested params extracted → let the attribute-style scan try.
        guard !arguments.isEmpty else { return nil }

        if let invalidArguments = schemaValidationFailure(
            toolName: funcName, arguments: arguments, tools: tools) {
            return ToolCall(function: .init(name: funcName, arguments: invalidArguments))
        }
        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }

    /// Qwen JSON wire form fallback: `<tool_call>{"name": …, "arguments": …}</tool_call>`.
    /// Strips the surrounding start/end tags (if present) and decodes the body as a
    /// `ToolCall.Function`, mirroring `JSONToolCallParser`. Returns nil unless the body
    /// is a JSON object so ordinary prose is never misread as a tool call.
    private func parseJSONToolCallFallback(_ content: String) -> ToolCall? {
        var text = content
        if let startTag, let r = text.range(of: startTag) {
            text = String(text[r.upperBound...])
        }
        if let endTag, let r = text.range(of: endTag) {
            text = String(text[..<r.lowerBound])
        }
        let jsonStr = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard jsonStr.hasPrefix("{"), jsonStr.hasSuffix("}"),
            let data = jsonStr.data(using: .utf8),
            let function = try? JSONDecoder().decode(ToolCall.Function.self, from: data)
        else { return nil }
        return ToolCall(function: function)
    }

    private func schemaValidationFailure(
        toolName: String,
        arguments: [String: any Sendable],
        tools: [[String: any Sendable]]?
    ) -> [String: any Sendable]? {
        guard let tools,
            let functionSpec = functionSpec(named: toolName, in: tools),
            let parameters = sendableObject(functionSpec["parameters"])
        else { return nil }

        for required in sendableStringArray(parameters["required"]) {
            if arguments[required] == nil {
                return invalidToolArguments(
                    toolName: toolName,
                    message: "missing required argument: \(required)",
                    field: required,
                    expected: "required parameter")
            }
        }

        return nil
    }

    private func functionSpec(
        named name: String,
        in tools: [[String: any Sendable]]
    ) -> [String: any Sendable]? {
        for tool in tools {
            let function = sendableObject(tool["function"]) ?? tool
            if function["name"] as? String == name {
                return function
            }
        }
        return nil
    }

    private func invalidToolArguments(
        toolName: String,
        message: String,
        field: String,
        expected: String
    ) -> [String: any Sendable] {
        [
            "_error": "invalid_tool_arguments",
            "_tool": toolName,
            "_message": message,
            "_field": field,
            "_expected": expected,
        ]
    }

    private func trimBoundaryNewlines(_ value: String) -> String {
        var result = value
        while result.hasPrefix("\n") || result.hasPrefix("\r") {
            result = String(result.dropFirst())
        }
        while result.hasSuffix("\n") || result.hasSuffix("\r") {
            result = String(result.dropLast())
        }
        return result
    }

    private func sendableObject(_ value: (any Sendable)?) -> [String: any Sendable]? {
        if let object = value as? [String: any Sendable] {
            return object
        }
        if let object = value as? NSDictionary {
            return sendableNSDictionary(object)
        }
        if case .object(let object)? = value as? JSONValue {
            return object.mapValues { $0.sendableValue }
        }
        return nil
    }

    private func sendableFoundationJSONValue(_ value: Any) -> (any Sendable)? {
        if let string = value as? String {
            return string
        }
        if let number = value as? NSNumber {
            return number
        }
        if let object = value as? NSDictionary {
            return sendableNSDictionary(object)
        }
        if let array = value as? NSArray {
            return array.compactMap { sendableFoundationJSONValue($0) } as [any Sendable]
        }
        if value is NSNull {
            return nil
        }
        return nil
    }

    private func sendableNSDictionary(_ object: NSDictionary) -> [String: any Sendable] {
        var out: [String: any Sendable] = [:]
        for (key, child) in object {
            guard let key = key as? String else { continue }
            if let sendable = sendableFoundationJSONValue(child) {
                out[key] = sendable
            }
        }
        return out
    }

    private func sendableStringArray(_ value: (any Sendable)?) -> [String] {
        if let strings = value as? [String] {
            return strings
        }
        if let values = value as? [any Sendable] {
            return values.compactMap { $0 as? String }
        }
        if let values = value as? NSArray {
            return values.compactMap { $0 as? String }
        }
        if case .array(let values)? = value as? JSONValue {
            return values.compactMap {
                if case .string(let value) = $0 { return value }
                return nil
            }
        }
        return []
    }

    private func decodeHTMLLineBreaks(_ value: String) -> String {
        value.replacingOccurrences(
            of: #"<br\s*/?>"#,
            with: "\n",
            options: [.regularExpression, .caseInsensitive])
    }
}

private func decodeQuotedStringParameter(_ value: String) -> String? {
    let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
    guard trimmed.hasPrefix("\""), trimmed.hasSuffix("\"") else {
        return nil
    }
    if let data = trimmed.data(using: .utf8),
       let decoded = try? JSONDecoder().decode(String.self, from: data) {
        return decoded
    }
    let innerStart = trimmed.index(after: trimmed.startIndex)
    let innerEnd = trimmed.index(before: trimmed.endIndex)
    guard innerStart <= innerEnd else { return "" }
    let inner = String(trimmed[innerStart..<innerEnd])
    if inner.contains("\n") || inner.contains("\r") {
        return inner
    }
    return nil
}
