import Foundation

/// K2-Horizon native IFM transport. A group commits atomically: a malformed
/// or truncated member never becomes an executable partial call.
public struct K2HorizonToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<ifm|tool_calls>"
    public let endTag: String? = "</ifm|tool_calls>"
    public var startTagAliases: [String] { [startTag!, "<ifm|tool_call>"] }
    public var endTagAliases: [String] { [endTag!, "</ifm|tool_call>"] }
    public var usesCustomEndBoundary: Bool { true }
    public var preservesWhitespaceBeforeToolCalls: Bool { true }
    public var orphanStripTags: [String] {
        [
            "</ifm|tool_calls>", "</ifm|tool_call>", "<ifm|arg_key>", "</ifm|arg_key>",
            "<ifm|arg_type>", "</ifm|arg_type>", "<ifm|arg_value>", "</ifm|arg_value>",
        ]
    }
    public init() {}

    public func completeToolCallEnd(in content: String) -> String.Index? {
        // JSON strings and native argument values own their literal bytes.
        let group = content.hasPrefix(startTag!)
        let closer = group ? endTag! : "</ifm|tool_call>"
        var i = content.startIndex
        var quoted = false
        var escaped = false
        while i < content.endIndex {
            let tail = content[i...]
            if quoted {
                if escaped {
                    escaped = false
                } else if content[i] == "\\" {
                    escaped = true
                } else if content[i] == "\"" {
                    quoted = false
                }
            } else if tail.hasPrefix("<ifm|arg_value>") {
                guard let end = content.range(of: "</ifm|arg_value>", range: i ..< content.endIndex)
                else { return nil }
                i = end.upperBound
                continue
            } else if tail.hasPrefix(closer) {
                return content.index(i, offsetBy: closer.count)
            } else if content[i] == "\"" {
                quoted = true
            }
            i = content.index(after: i)
        }
        return nil
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        parseEOS(content, tools: tools).first
    }

    public func parseEOS(_ content: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        var text = content.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.hasPrefix(startTag!) {
            guard let end = completeToolCallEnd(in: text), end == text.endIndex else { return [] }
            text = String(text.dropFirst(startTag!.count).dropLast(endTag!.count))
            // Native low-effort K2 may repeat the group opener before its first
            // member. One closing envelope terminates that prefix. Only consume
            // consecutive leading openers: never rewrite argument values or
            // accept a new nested envelope after a call has started.
            text = text.trimmingCharacters(in: .whitespacesAndNewlines)
            while text.hasPrefix(startTag!) {
                text = String(text.dropFirst(startTag!.count))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        var calls: [ToolCall] = []
        while !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            text = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard text.hasPrefix("<ifm|tool_call>"),
                let end = completeToolCallEnd(in: text),
                let call = parseBody(String(text[..<end].dropFirst(15).dropLast(16)), tools: tools)
            else { return [] }
            calls.append(call)
            text = String(text[end...])
        }
        return calls
    }

    private func parseBody(_ body: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("{") {
            guard
                let object = try? JSONSerialization.jsonObject(with: Data(trimmed.utf8))
                    as? [String: Any],
                let name = object["name"] as? String, validName(name),
                let args = object["arguments"] as? [String: Any]
            else { return nil }
            return ToolCall(function: .init(name: name, arguments: args.mapValues(asSendable)))
        }
        let name = String(trimmed.prefix { !$0.isWhitespace && $0 != "<" })
        guard validName(name) else { return nil }
        var rest = String(trimmed.dropFirst(name.count))
        var args: [String: any Sendable] = [:]
        var order: [String] = []
        func take(_ tag: String, from text: inout String) -> String? {
            text = text.trimmingCharacters(in: .whitespacesAndNewlines)
            let open = "<ifm|\(tag)>"
            let close = "</ifm|\(tag)>"
            guard text.hasPrefix(open), let end = text.range(of: close) else { return nil }
            let value = String(
                text[text.index(text.startIndex, offsetBy: open.count) ..< end.lowerBound])
            text = String(text[end.upperBound...])
            return value
        }
        while !rest.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            guard let rawKey = take("arg_key", from: &rest) else { return nil }
            let key = rawKey.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !key.isEmpty, !key.contains("<"), args[key] == nil else { return nil }
            let declared =
                rest.trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("<ifm|arg_type>")
                ? take("arg_type", from: &rest)?.trimmingCharacters(in: .whitespacesAndNewlines)
                : nil
            guard let value = take("arg_value", from: &rest) else { return nil }
            let normalizedTools = tools?.map { tool -> [String: any Sendable] in
                tool["function"] == nil ? ["function": tool] : tool
            }
            let schema = getParameterSchema(funcName: name, paramName: key, tools: normalizedTools)
            // The template prints array[item-type], not just array. Schema
            // string alternatives remain authoritative over emitted type hints.
            let declaredType = declared.map {
                $0.hasPrefix("array[") && $0.hasSuffix("]") ? "array" : $0
            }
            let types =
                schema.map { extractTypesFromSchema($0) }
                ?? declaredType.map { [$0] } ?? ["string"]
            if types.contains("string") {
                args[key] = value
            } else {
                guard
                    let decoded = try? JSONSerialization.jsonObject(
                        with: Data(value.utf8), options: [.fragmentsAllowed])
                else { return nil }
                let literal = value.trimmingCharacters(in: .whitespacesAndNewlines)
                let valid = types.contains { type in
                    switch type {
                    case "boolean": return literal == "true" || literal == "false"
                    case "null": return decoded is NSNull
                    case "integer": return Int(literal) != nil
                    case "number": return Double(literal).map { $0.isFinite } == true
                    case "array": return decoded is [Any]
                    case "object": return decoded is [String: Any]
                    case "any": return true
                    default: return false
                    }
                }
                guard valid else { return nil }
                args[key] = asSendable(decoded)
            }
            order.append(key)
        }
        return ToolCall(function: .init(name: name, arguments: args, argumentOrder: order))
    }

    private func validName(_ name: String) -> Bool {
        !name.isEmpty && name.allSatisfy { $0.isLetter || $0.isNumber || "_.-".contains($0) }
    }
}
