// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for XML function format: <function=name><parameter=key>value</parameter></function>
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/qwen3_coder.py
public struct XMLFunctionParser: ToolCallParser, Sendable {
    public let startTag: String?
    public let endTag: String?
    public let decodesHTMLLineBreaks: Bool

    public init(startTag: String, endTag: String, decodesHTMLLineBreaks: Bool = false) {
        self.startTag = startTag
        self.endTag = endTag
        self.decodesHTMLLineBreaks = decodesHTMLLineBreaks
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        // Pattern: <function=(content)</function> — [\s\S] matches newlines
        guard
            let funcMatch = content.range(
                of: #"<function=([\s\S]*?)</function>"#, options: .regularExpression)
        else { return nil }

        let funcContent = String(content[funcMatch])

        // Extract function name (everything between <function= and first >)
        guard let nameStart = funcContent.range(of: "<function="),
            let nameEnd = funcContent.range(
                of: ">", range: nameStart.upperBound ..< funcContent.endIndex)
        else { return nil }

        let funcName = String(funcContent[nameStart.upperBound ..< nameEnd.lowerBound])
        let paramSection = String(funcContent[nameEnd.upperBound...])

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

            // Find the closing </parameter> tag
            guard
                let paramEnd = paramSection.range(
                    of: "</parameter>", range: nameEnd.upperBound ..< paramSection.endIndex)
            else { break }

            var paramValue = String(paramSection[nameEnd.upperBound ..< paramEnd.lowerBound])

            // Trim leading/trailing newlines (matching Python behavior)
            if paramValue.hasPrefix("\n") {
                paramValue = String(paramValue.dropFirst())
            }
            if paramValue.hasSuffix("\n") {
                paramValue = String(paramValue.dropLast())
            }

            if decodesHTMLLineBreaks,
               isStringType(funcName: funcName, argName: paramName, tools: tools) {
                paramValue = decodeHTMLLineBreaks(paramValue)
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
