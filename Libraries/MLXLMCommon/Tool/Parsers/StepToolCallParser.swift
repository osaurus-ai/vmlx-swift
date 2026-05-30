// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation

/// Parser for StepFun Step 3.5 / 3.7 tool calls.
///
/// Step's documented template is the Qwen-style XML function envelope, but
/// live Step 3.7 rows can emit a bare registered function call with a single
/// JSON argument object while the reasoning rail is open:
///
///     line_count({"text":"red\ngreen\nblue"})
///
/// Keep this fallback narrow: it only accepts registered tool names, validates
/// required arguments, and rejects unknown arguments when the schema declares
/// properties.
public struct StepToolCallParser: ToolCallParser, Sendable {
    private let xml = XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")

    public let startTag: String? = "<tool_call>"
    public let endTag: String? = "</tool_call>"
    public let supportsInlineJSONToolFallback = true

    public var startTagAliases: [String] { xml.startTagAliases }
    public var endTagAliases: [String] { xml.endTagAliases }
    public var startTagPrefixes: [String] { xml.startTagPrefixes }
    public var endTagPrefixes: [String] { xml.endTagPrefixes }

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        if let call = xml.parse(content: content, tools: tools) {
            return call
        }
        return parseBareFunctionJSON(content, tools: tools)
    }

    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        let xmlCalls = xml.parseEOS(toolCallBuffer, tools: tools)
        if !xmlCalls.isEmpty {
            return xmlCalls
        }
        guard let call = parseBareFunctionJSON(toolCallBuffer, tools: tools) else {
            return []
        }
        return [call]
    }

    private func parseBareFunctionJSON(
        _ content: String,
        tools: [[String: any Sendable]]?
    ) -> ToolCall? {
        let text = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let open = text.firstIndex(of: "("),
            let close = text.lastIndex(of: ")"),
            close > open
        else { return nil }

        let name = String(text[..<open])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty,
            name.allSatisfy({ $0.isLetter || $0.isNumber || $0 == "_" }),
            let spec = toolSpec(named: name, tools: tools)
        else { return nil }

        let tail = text[text.index(after: close)...]
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard tail.isEmpty else { return nil }

        let rawArgs = String(text[text.index(after: open)..<close])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard rawArgs.hasPrefix("{"),
            let object = parseJSONObject(rawArgs)
        else { return nil }

        let args: [String: Any]
        if object.count == 1,
            let rawArguments = object["arguments"],
            let nested = firstArgumentObject(from: rawArguments)
        {
            args = nested
        } else {
            args = object
        }

        guard spec.required.allSatisfy({ args[$0] != nil }) else {
            return nil
        }
        if !spec.properties.isEmpty {
            guard args.keys.allSatisfy({ spec.properties.contains($0) }) else {
                return nil
            }
        }
        return ToolCall(function: .init(name: name, arguments: args.mapValues(asSendable)))
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

    private func toolSpec(
        named name: String,
        tools: [[String: any Sendable]]?
    ) -> (required: [String], properties: Set<String>)? {
        guard let tools else { return nil }
        for tool in tools {
            let function = (tool["function"] as? [String: any Sendable]) ?? tool
            guard function["name"] as? String == name else { continue }
            let parameters = function["parameters"] as? [String: any Sendable]
            let properties = parameters?["properties"] as? [String: any Sendable]
            return (
                parameters?["required"] as? [String] ?? [],
                properties.map { Set($0.keys) } ?? []
            )
        }
        return nil
    }
}
