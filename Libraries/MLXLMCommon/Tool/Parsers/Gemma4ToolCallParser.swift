// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// Gemma-4 tool-call parser.
///
/// Gemma-4 bundles normally use the native `<|tool_call>...<tool_call|>`
/// function-call envelope, but live JANG/CRACK rows can emit the same
/// Zyphra XML function envelope used by ZAYA:
///
/// `<zyphra_tool_call><function=name><parameter=key>...</parameter></function></zyphra_tool_call>`
///
/// Accept both transports as real tool-call protocol. This is parser
/// recognition only: it does not inject tools, rewrite prompts, change sampler
/// defaults, or normalize argument values beyond the XML parser's schema-aware
/// string handling.
public struct Gemma4ToolCallParser: ToolCallParser, Sendable {
    private let native = GemmaFunctionParser(
        startTag: "<|tool_call>",
        endTag: "<tool_call|>",
        escapeMarker: "<|\"|>")
    private let zyphra = XMLFunctionParser(
        startTag: "<zyphra_tool_call>",
        endTag: "</zyphra_tool_call>",
        decodesHTMLLineBreaks: true,
        unwrapJSONQuotedStringParameters: true)

    public init() {}

    public var startTag: String? { native.startTag }
    public var endTag: String? { native.endTag }
    public var supportsBareCallToolFallback: Bool { true }

    public var startTagAliases: [String] {
        // Live JANG/CRACK rows occasionally drift the native `<|tool_call>` open
        // delimiter to `<|tool_call|>` (an extra pipe mirroring the `<tool_call|>`
        // close) and wrap the arguments in parens, e.g.
        // `<|tool_call|>call:name({"k":"v"})<tool_call|>`. Recognize the drifted
        // open tag so the streaming processor buffers the envelope instead of
        // leaking the raw markup as visible content. The close `<tool_call|>` is
        // unchanged, so it still pairs through `endTagAliases`.
        ["<|tool_call|>"] + native.startTagAliases + zyphra.startTagAliases
    }

    public var endTagAliases: [String] {
        native.endTagAliases + zyphra.endTagAliases
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        native.parse(content: content, tools: tools)
            ?? zyphra.parse(content: content, tools: tools)
            ?? parseSchemaConstrainedNativeDrift(content: content, tools: tools)
    }

    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        let nativeCalls = native.parseEOS(toolCallBuffer, tools: tools)
        if !nativeCalls.isEmpty { return nativeCalls }
        let zyphraCalls = zyphra.parseEOS(toolCallBuffer, tools: tools)
        if !zyphraCalls.isEmpty { return zyphraCalls }
        guard let drift = parseSchemaConstrainedNativeDrift(content: toolCallBuffer, tools: tools) else {
            return []
        }
        return [drift]
    }

    private func parseSchemaConstrainedNativeDrift(
        content: String,
        tools: [[String: any Sendable]]?
    ) -> ToolCall? {
        guard let spec = singleToolSpec(tools: tools),
            let name = spec.name,
            !name.isEmpty,
            let braceStart = content.firstIndex(of: "{"),
            let braceEnd = content.lastIndex(of: "}"),
            braceStart < braceEnd
        else { return nil }

        let rawName = content[..<braceStart]
            .replacingOccurrences(of: "<|tool_call>", with: "")
            .replacingOccurrences(of: "call:", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard resolvesSchemaDrift(rawName, to: name) else { return nil }

        var args = String(content[content.index(after: braceStart) ..< braceEnd])
        if let properties = spec.properties, !properties.isEmpty {
            args = rewriteArgumentKeys(args, allowed: properties)
        }

        let corrected = "call:\(name){\(args)}"
        guard let call = native.parse(content: corrected, tools: tools),
            call.function.name == name
        else { return nil }
        return call
    }

    private func rewriteArgumentKeys(_ args: String, allowed: Set<String>) -> String {
        var output = ""
        var cursor = args.startIndex

        while cursor < args.endIndex {
            guard let colon = args[cursor...].firstIndex(of: ":") else {
                output += args[cursor...]
                break
            }

            let keyStart = output.isEmpty ? cursor : args[cursor...].firstNonSeparatorIndex ?? cursor
            output += args[cursor..<keyStart]

            let rawKey = String(args[keyStart..<colon]).trimmingCharacters(in: .whitespacesAndNewlines)
            if let resolved = allowed.first(where: { resolvesSchemaDrift(rawKey, to: $0) }) {
                output += resolved
            } else {
                output += args[keyStart..<colon]
            }
            output.append(":")
            cursor = args.index(after: colon)

            while cursor < args.endIndex {
                let ch = args[cursor]
                output.append(ch)
                cursor = args.index(after: cursor)
                if ch == "," { break }
            }
        }

        return output
    }

    private func resolvesSchemaDrift(_ raw: String, to expected: String) -> Bool {
        let value = raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "`\"'"))
        guard !value.isEmpty else { return false }
        if value == expected { return true }
        if value.replacingOccurrences(of: "c", with: "")
            == expected.replacingOccurrences(of: "c", with: "")
        {
            return true
        }
        return editDistanceAtMostOne(value, expected)
    }

    private func editDistanceAtMostOne(_ lhs: String, _ rhs: String) -> Bool {
        let left = Array(lhs)
        let right = Array(rhs)
        if abs(left.count - right.count) > 1 { return false }

        var i = 0
        var j = 0
        var edits = 0
        while i < left.count && j < right.count {
            if left[i] == right[j] {
                i += 1
                j += 1
                continue
            }
            edits += 1
            if edits > 1 { return false }
            if left.count > right.count {
                i += 1
            } else if right.count > left.count {
                j += 1
            } else {
                i += 1
                j += 1
            }
        }
        if i < left.count || j < right.count { edits += 1 }
        return edits <= 1
    }

    private func singleToolSpec(
        tools: [[String: any Sendable]]?
    ) -> (name: String?, properties: Set<String>?)? {
        guard let tools, tools.count == 1 else { return nil }
        let function = (tools[0]["function"] as? [String: any Sendable]) ?? tools[0]
        let parameters = function["parameters"] as? [String: any Sendable]
        let properties = parameters?["properties"] as? [String: any Sendable]
        return (function["name"] as? String, properties.map { Set($0.keys) })
    }
}

private extension Substring {
    var firstNonSeparatorIndex: Index? {
        firstIndex { ch in
            ch != "," && !ch.isWhitespace && !ch.isNewline
        }
    }
}
