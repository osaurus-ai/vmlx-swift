// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// ERNIE wraps one or more invoke records in a single tool-call block.
public struct Ernie55ToolCallParser: ToolCallParser {
    public let startTag: String? = "<ernie:tool_call>"
    public let endTag: String? = "</ernie:tool_call>"

    public init() {}

    public var orphanStripTags: [String] { ["</ernie:tool_call>", "</invoke>", "</parameter>"] }

    public func parse(content: String, tools: [ToolSpec]?) -> ToolCall? {
        parseEOS(content, tools: tools).first
    }

    public func parseEOS(_ content: String, tools: [ToolSpec]?) -> [ToolCall] {
        let input = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let startTag, let endTag, input.hasPrefix(startTag), input.hasSuffix(endTag) else { return [] }
        var body = input.dropFirst(startTag.count).dropLast(endTag.count)
        var calls: [ToolCall] = []
        while !body.isEmpty {
            consumeWhitespace(&body)
            if body.isEmpty { break }
            guard let name = consumeName(&body, prefix: "<invoke name=\"") else { return [] }
            let declarations = tools?.map { $0["function"] as? ToolSpec ?? $0 }
            if let declarations, !declarations.contains(where: { $0["name"] as? String == name }) { return [] }
            let properties = getParameterConfig(funcName: name, tools: tools)
            var arguments: [String: any Sendable] = [:]
            while true {
                consumeWhitespace(&body)
                if body.hasPrefix("</invoke>") {
                    body = body.dropFirst("</invoke>".count)
                    break
                }
                guard let key = consumeName(&body, prefix: "<parameter name=\""),
                    arguments[key] == nil, let closer = body.range(of: "</parameter>")
                else { return [] }
                let value = String(body[..<closer.lowerBound])
                let types = extractTypesFromSchema(properties[key] as? ToolSpec)
                arguments[key] = types == ["string"] ? value : convertValueWithTypes(value, types: types)
                body = body[closer.upperBound...]
            }
            calls.append(ToolCall(function: .init(name: name, arguments: arguments)))
        }
        return calls
    }

    private func consumeWhitespace(_ text: inout Substring) {
        text = text.drop(while: \.isWhitespace)
    }

    private func consumeName(_ text: inout Substring, prefix: String) -> String? {
        guard text.hasPrefix(prefix),
            let delimiter = text.range(of: "\">", range: text.index(text.startIndex, offsetBy: prefix.count)..<text.endIndex)
        else { return nil }
        let name = String(text.dropFirst(prefix.count)[..<delimiter.lowerBound])
        guard !name.isEmpty, !name.contains("\""), !name.contains("<") else { return nil }
        text = text[delimiter.upperBound...]
        return name
    }
}
