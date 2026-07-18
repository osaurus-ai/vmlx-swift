// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// MiniMax-M3 tool-call parser — a faithful port of vllm-mlx
// `tool_parsers/minimax_m3_tool_parser.py` (the authoritative M3 format).
//
// M3 emits Anthropic-style XML where the PARAMETER NAME IS THE TAG (not a
// `name=` attribute), wrapped in `<tool_call>...</tool_call>`, with the literal
// namespace token `]<]minimax[>[` prefixed before every element, and reasoning in
// `<mm:think>...</mm:think>`. Rendered shape (ns shown as · , stripped first):
//
//     <tool_call>
//     ·<invoke name="get_weather">
//     ·<location>San Francisco·</location>
//     ·<opts>·<unit>celsius·</unit>·</opts>          (nested object)
//     ·<days>·<item>mon·</item>·<item>tue·</item>·</days>  (array)
//     ·</invoke>
//     </tool_call>
//
// Value mapping:  scalar -> coerced (JSON-parsed else string); `<item>` repeats ->
// array; other nested tags -> object. This is NOT minimax_m2's `<minimax:tool_call>`
// + `<parameter name=>` form, so M3 needs its own parser/route.

import Foundation

public struct MiniMaxM3ToolCallParser: ToolCallParser, Sendable {
    private static let ns = "]<]minimax[>["

    public let startTag: String? = "<tool_call>"
    public let endTag: String? = "</tool_call>"
    /// The model prefixes the namespace token before the envelope; accept it too.
    public var startTagAliases: [String] { ["\(MiniMaxM3ToolCallParser.ns)<tool_call>"] }
    public var endTagAliases: [String] { ["\(MiniMaxM3ToolCallParser.ns)</tool_call>"] }

    public init() {}

    // MARK: noise stripping (ns token + reasoning block)

    private func stripNoise(_ text: String) -> String {
        var t = text.replacingOccurrences(of: Self.ns, with: "")
        // Remove <mm:think>...</mm:think> blocks (DOTALL).
        while let open = t.range(of: "<mm:think>"),
            let close = t.range(of: "</mm:think>", range: open.upperBound ..< t.endIndex) {
            t.removeSubrange(open.lowerBound ..< close.upperBound)
        }
        return t
    }

    // MARK: recursive tag parsing (mirrors _next_tag / _children / _parse_value)

    /// Next top-level `<tag>...</tag>` at/after `pos`, with correct nesting of
    /// same-named tags. Returns (tag, inner, endIndex) or nil if none complete.
    private func nextTag(_ s: String, _ pos: String.Index)
        -> (tag: String, inner: String, end: String.Index)? {
        guard let openLt = s.range(of: "<", range: pos ..< s.endIndex),
            let openGt = s.range(of: ">", range: openLt.upperBound ..< s.endIndex)
        else { return nil }
        let rawTag = String(s[openLt.upperBound ..< openGt.lowerBound])
            .trimmingCharacters(in: .whitespaces)
        // Skip closing tags / attributes-bearing openers (only bare `<tag>` here).
        guard !rawTag.isEmpty, !rawTag.hasPrefix("/"),
            rawTag.allSatisfy({ $0.isLetter || $0.isNumber || "_-.:".contains($0) })
        else {
            return nextTag(s, openGt.upperBound)
        }
        let openTag = "<\(rawTag)>"
        let closeTag = "</\(rawTag)>"
        var depth = 1
        var scan = openGt.upperBound
        while true {
            let nextOpen = s.range(of: openTag, range: scan ..< s.endIndex)
            guard let nextClose = s.range(of: closeTag, range: scan ..< s.endIndex) else {
                return nil  // unterminated
            }
            if let nextOpen, nextOpen.lowerBound < nextClose.lowerBound {
                depth += 1
                scan = nextOpen.upperBound
            } else {
                depth -= 1
                scan = nextClose.upperBound
                if depth == 0 {
                    return (rawTag, String(s[openGt.upperBound ..< nextClose.lowerBound]), scan)
                }
            }
        }
    }

    private func children(_ s: String) -> [(tag: String, inner: String)] {
        var out: [(String, String)] = []
        var pos = s.startIndex
        while let nxt = nextTag(s, pos) {
            out.append((nxt.tag, nxt.inner))
            pos = nxt.end
        }
        return out
    }

    private func coerce(_ value: String) -> any Sendable {
        let v = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if v.isEmpty { return v }
        if v == "true" { return true }
        if v == "false" { return false }
        if v == "null" { return NSNull() }
        if let i = Int(v) { return i }
        if let d = Double(v) { return d }
        // Nested objects/arrays arrive as XML tags (handled by parseValue), not
        // JSON-in-a-leaf, so a leaf value is returned as its coerced scalar/string.
        return v
    }

    private func parseValue(_ inner: String) -> any Sendable {
        let kids = children(inner)
        if kids.isEmpty { return coerce(inner) }
        if kids.allSatisfy({ $0.tag == "item" }) {
            return kids.map { parseValue($0.inner) } as [any Sendable]
        }
        var obj: [String: any Sendable] = [:]
        for (tag, ci) in kids { obj[tag] = parseValue(ci) }
        return obj
    }

    private func argsFromInvoke(_ body: String) -> [String: any Sendable] {
        var args: [String: any Sendable] = [:]
        for (tag, inner) in children(body) { args[tag] = parseValue(inner) }
        return args
    }

    // MARK: ToolCallParser

    public func isValidPartialContent(_ toolCallBuffer: String) -> Bool {
        let cleaned = stripNoise(toolCallBuffer)
        guard !cleaned.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return true }
        // Building toward a <tool_call> or <invoke ...> opener.
        return cleaned.contains("<tool_call")
            || cleaned.contains("<invoke")
            || "<tool_call>".hasPrefix(cleaned)
            || "<invoke".hasPrefix(cleaned)
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        let cleaned = stripNoise(content)
        guard cleaned.contains("<invoke") else { return nil }

        // Search inside <tool_call>...</tool_call> if present; else whole text
        // (covers truncation / wrapper-less invoke).
        var searchSpace = cleaned
        if let open = cleaned.range(of: "<tool_call>") {
            let afterOpen = open.upperBound
            if let close = cleaned.range(of: "</tool_call>", range: afterOpen ..< cleaned.endIndex) {
                searchSpace = String(cleaned[afterOpen ..< close.lowerBound])
            } else {
                searchSpace = String(cleaned[afterOpen...])  // truncated, no closer
            }
        }

        // First <invoke name=...>...</invoke>. Name may be "x" / 'x' / bare.
        guard let invokeOpen = searchSpace.range(of: "<invoke") else { return nil }
        guard let nameEq = searchSpace.range(
            of: "name=", range: invokeOpen.upperBound ..< searchSpace.endIndex)
        else { return nil }
        guard let headerEnd = searchSpace.range(
            of: ">", range: nameEq.upperBound ..< searchSpace.endIndex)
        else { return nil }
        let rawName = String(searchSpace[nameEq.upperBound ..< headerEnd.lowerBound])
            .trimmingCharacters(in: .whitespaces)
        let funcName = rawName.trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
        guard !funcName.isEmpty else { return nil }

        guard let invokeClose = searchSpace.range(
            of: "</invoke>", range: headerEnd.upperBound ..< searchSpace.endIndex)
        else { return nil }
        let body = String(searchSpace[headerEnd.upperBound ..< invokeClose.lowerBound])

        return ToolCall(function: .init(name: funcName, arguments: argsFromInvoke(body)))
    }
}
