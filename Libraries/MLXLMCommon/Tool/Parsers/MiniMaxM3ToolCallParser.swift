// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// MiniMax-M3 / M3.1 tool calls (`ToolCallFormat.minimaxM3`).
///
/// The M3 chat template writes every tool-call element behind the namespace separator token `]<]minimax[>[` and names
/// each argument by its TAG (not a `name=` attribute):
///
///     ]<]minimax[>[<tool_call>
///     ]<]minimax[>[<invoke name="get_weather">
///     ]<]minimax[>[<location>Paris]<]minimax[>[</location>
///     ]<]minimax[>[<opts>]<]minimax[>[<unit>celsius]<]minimax[>[</unit>]<]minimax[>[</opts>
///     ]<]minimax[>[<days>]<]minimax[>[<item>mon]<]minimax[>[</item>]<]minimax[>[<item>tue]<]minimax[>[</item>]<]minimax[>[</days>
///     ]<]minimax[>[</invoke>
///     </tool_call>
///
/// One `<tool_call>` block may hold several `<invoke>` elements (each becomes a call). Values: an element with no
/// child elements is a scalar (the parameter's schema type when the tool declares one, else JSON when it parses as
/// JSON, else the trimmed text); an element whose children are all `<item>` is an array; any other element with
/// children is an object keyed by the child tags. This is the vMLX Python `minimax_m3` parser's contract.
///
/// Streaming: the namespace token opens collection (the detokenizer keeps special tokens, so it precedes
/// `<tool_call>` in the text stream) and `</tool_call>` closes it. A separator truncated by its last `[` before
/// `<tool_call>` (seen in live M3 output) is accepted too. Namespace tokens never reach visible text from a parsed
/// envelope.
public struct MiniMaxM3ToolCallParser: ToolCallParser, Sendable {
    public static let namespaceToken = "]<]minimax[>["
    static let truncatedNamespaceToken = String(namespaceToken.dropLast())

    public let startTag: String? = MiniMaxM3ToolCallParser.namespaceToken
    public let endTag: String? = "</tool_call>"

    public var startTagAliases: [String] {
        [Self.namespaceToken, Self.truncatedNamespaceToken + "<tool_call>"]
    }

    public init() {}

    /// The collected envelope (namespace tokens removed) must still be able to become `<tool_call>` or `<invoke`.
    public func isValidPartialContent(_ toolCallBuffer: String) -> Bool {
        let body = Self.stripNamespace(toolCallBuffer).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !body.isEmpty else { return true }
        return ["<tool_call>", "<invoke"].contains { $0.hasPrefix(body) || body.hasPrefix($0) }
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        parseEOS(content, tools: tools).first
    }

    /// Every `<invoke>` of the envelope, in order.
    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        let text = Self.stripNamespace(toolCallBuffer)
        var calls: [ToolCall] = []
        var cursor = text.startIndex
        while let open = text.range(of: "<invoke", range: cursor ..< text.endIndex),
            let headEnd = text.range(of: ">", range: open.upperBound ..< text.endIndex),
            let close = text.range(of: "</invoke>", range: headEnd.upperBound ..< text.endIndex)
        {
            cursor = close.upperBound
            let head = text[open.upperBound ..< headEnd.lowerBound].trimmingCharacters(in: .whitespaces)
            guard head.hasPrefix("name") else { continue }
            let name = extractName(
                String(head.dropFirst("name".count)).trimmingCharacters(in: .whitespaces)
                    .drop(while: { $0 == "=" }).trimmingCharacters(in: .whitespaces))
            guard !name.isEmpty else { continue }
            let schema = getParameterConfig(funcName: name, tools: tools)
            var arguments: [String: any Sendable] = [:]
            for (tag, inner) in Self.children(of: text[headEnd.upperBound ..< close.lowerBound]) {
                // A declared parameter takes its schema type; an undeclared one is coerced like a nested value.
                let declared = (schema[tag] as? [String: any Sendable]).map { extractTypesFromSchema($0) }
                arguments[tag] = Self.value(inner, types: declared ?? [])
            }
            calls.append(ToolCall(function: .init(name: name, arguments: arguments)))
        }
        return calls
    }

    static func stripNamespace(_ text: String) -> String {
        var out = text.replacingOccurrences(of: namespaceToken, with: "")
        out = out.replacingOccurrences(of: truncatedNamespaceToken, with: "")
        return out
    }

    /// A parameter value: structured when it has child elements, else a scalar (schema type, JSON, or text).
    static func value(_ inner: Substring, types: [String] = []) -> any Sendable {
        let kids = children(of: inner)
        if kids.isEmpty {
            let text = inner.trimmingCharacters(in: .whitespacesAndNewlines)
            if text.isEmpty { return text }
            if !types.isEmpty { return convertValueWithTypes(text, types: types) }
            return jsonScalarOrText(text)
        }
        if kids.allSatisfy({ $0.tag == "item" }) {
            return kids.map { value($0.inner) }
        }
        var object: [String: any Sendable] = [:]
        for (tag, kidInner) in kids { object[tag] = value(kidInner) }  // a repeated key keeps the last value
        return object
    }

    /// JSON when the text parses as JSON (including a bare number, boolean, null or quoted string), else the text.
    static func jsonScalarOrText(_ text: String) -> any Sendable {
        guard let data = text.data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        else { return text }
        return asSendable(object)
    }

    /// The top-level `<tag>…</tag>` elements of `text`, in order; same-named nested elements are matched by depth.
    /// An element without its close ends the scan.
    static func children(of text: Substring) -> [(tag: String, inner: Substring)] {
        var out: [(tag: String, inner: Substring)] = []
        var cursor = text.startIndex
        while let (tag, openEnd) = nextOpenTag(in: text, from: cursor) {
            let open = "<\(tag)>", close = "</\(tag)>"
            var depth = 1
            var scan = openEnd
            var closed: Range<Substring.Index>?
            while depth > 0, let nextClose = text.range(of: close, range: scan ..< text.endIndex) {
                if let nextOpen = text.range(of: open, range: scan ..< nextClose.lowerBound) {
                    depth += 1
                    scan = nextOpen.upperBound
                } else {
                    depth -= 1
                    scan = nextClose.upperBound
                    if depth == 0 { closed = nextClose }
                }
            }
            guard let closed else { break }
            out.append((tag, text[openEnd ..< closed.lowerBound]))
            cursor = closed.upperBound
        }
        return out
    }

    /// The next opening element `<name>` (name: a letter or `_`, then letters, digits, `_`, `-`, `.`, `:`; optional
    /// whitespace before `>`) at or after `from`: its name and the index after `>`.
    private static func nextOpenTag(in text: Substring, from: Substring.Index)
        -> (String, Substring.Index)?
    {
        var search = from
        while let lt = text[search...].firstIndex(of: "<") {
            var i = text.index(after: lt)
            guard i < text.endIndex, let first = text[i].unicodeScalars.first,
                CharacterSet.letters.contains(first) || text[i] == "_"
            else {
                search = text.index(after: lt)
                continue
            }
            var name = ""
            while i < text.endIndex, text[i].isLetter || text[i].isNumber || "_-.:".contains(text[i]) {
                name.append(text[i])
                i = text.index(after: i)
            }
            while i < text.endIndex, text[i] == " " || text[i] == "\t" || text[i] == "\n" { i = text.index(after: i) }
            if i < text.endIndex, text[i] == ">" {
                return (name, text.index(after: i))
            }
            search = text.index(after: lt)
        }
        return nil
    }
}
