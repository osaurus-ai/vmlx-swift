// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DSML (DeepSeek Markup Language) tool-call parser. Format used by
// DeepSeek-V4-Flash / -Pro bundles per
// `jang/research/DSV-FAMILY-RUNTIME-GUIDE.md` §24.
//
// Example model output:
//
//     <｜DSML｜tool_calls>
//     <｜DSML｜invoke name="get_weather">
//     <｜DSML｜parameter name="location" string="true">San Francisco</｜DSML｜parameter>
//     <｜DSML｜parameter name="units" string="true">celsius</｜DSML｜parameter>
//     </｜DSML｜invoke>
//     <｜DSML｜invoke name="get_time">
//     <｜DSML｜parameter name="timezone" string="true">America/Los_Angeles</｜DSML｜parameter>
//     </｜DSML｜invoke>
//     </｜DSML｜tool_calls>
//
// CRITICAL: the `｜` are CURLY QUOTES (fullwidth vertical bar, U+FF5C),
// NOT ASCII pipe `|`. Pasting the literal characters into a Swift
// string works as long as the file is UTF-8 (default).
//
// Parameter encoding (per §24):
//   - string="true"  — value is a plain string, use as-is
//   - string="false" — value is JSON (int, bool, float, array, object)
//
// Block tokens:
//   outer:     <｜DSML｜tool_calls>      ... </｜DSML｜tool_calls>
//   per-call:  <｜DSML｜invoke name="..."> ... </｜DSML｜invoke>
//   per-param: <｜DSML｜parameter name="..." string="true|false"> ... </｜DSML｜parameter>
//
// Tool-result responses arrive from the user side as:
//   <tool_result>{JSON}</tool_result>
//
// (We don't parse tool results here — that's a render-side concern
// when echoing tool outputs back into the next turn's chat history.)

import Foundation

public struct DSMLToolCallParser: ToolCallParser, Sendable {
    // Curly-quote pipe U+FF5C.
    static let dsmlPrefix = "<\u{FF5C}DSML\u{FF5C}"
    static let dsmlPrefixClose = "</\u{FF5C}DSML\u{FF5C}"
    static let dsmlToolStartPrefix = "<\u{FF5C}DSML\u{FF5C}tool_"
    static let dsmlToolEndPrefix = "</\u{FF5C}DSML\u{FF5C}tool_"

    public let startTag: String? = "<\u{FF5C}DSML\u{FF5C}tool_calls>"
    public let endTag: String? = "</\u{FF5C}DSML\u{FF5C}tool_calls>"
    public let startTagAliases: [String] = [
        "<\u{FF5C}DSML\u{FF5C}tool_calls>",
        // Live DeepSeek V4 Flash sometimes drops the second "l".
        "<\u{FF5C}DSML\u{FF5C}tool_cals>",
        // Live DSV4 JANGTQ2 app decode has also emitted an extra "c"
        // after the underscore while preserving a valid invoke body.
        "<\u{FF5C}DSML\u{FF5C}tool_ccalls>",
        // Live DSV4 can also drift the suffix to "crs" while preserving
        // the DSML invoke/parameter body.
        "<\u{FF5C}DSML\u{FF5C}tool_crs>",
    ]
    public let endTagAliases: [String] = [
        "</\u{FF5C}DSML\u{FF5C}tool_calls>",
        "</\u{FF5C}DSML\u{FF5C}tool_cals>",
        // Same live alias family as `tool_ccalls`, abbreviated at the
        // suffix rather than the middle of the token.
        "</\u{FF5C}DSML\u{FF5C}tool_cs>",
        "</\u{FF5C}DSML\u{FF5C}tool_crs>",
    ]
    public let startTagPrefixes: [String] = [Self.dsmlToolStartPrefix]
    public let endTagPrefixes: [String] = [Self.dsmlToolEndPrefix]
    public let supportsInlineJSONToolFallback = true

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        // Strip outer block if present.
        let text = strippedOuterToolTags(from: content)

        // Find first <｜DSML｜invoke name="...">
        if let firstCall = parseFirstInvoke(in: text, tools: tools) {
            return firstCall
        }
        if let pythonCall = parsePythonStyleToolFallback(in: text, tools: tools) {
            return pythonCall
        }
        if let bareNameJSONCall = parseBareNameJSONToolFallback(
            in: text,
            tools: tools,
            allowMalformedEOF: false)
        {
            return bareNameJSONCall
        }
        if let bareNameKeyValueCall = parseBareNameKeyValueToolFallback(
            in: text,
            tools: tools,
            allowEOF: true)
        {
            return bareNameKeyValueCall
        }
        if let actionJSONCall = parseActionJSONToolFallback(in: text, tools: tools) {
            return actionJSONCall
        }
        return parseInlineJSONToolFallback(in: text, tools: tools)
    }

    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall]
    {
        // DSML can carry multiple invokes within a single
        // <｜DSML｜tool_calls> block, so the default `components(by:
        // startTag)` strategy won't round-trip — each multi-invoke
        // block is ONE `startTag..endTag` outer envelope, split
        // internally by `<｜DSML｜invoke ...>` per call. Parse all
        // invokes in order.
        let buffer = strippedOuterToolTags(from: toolCallBuffer)
        let calls = parseAllInvokes(in: buffer, tools: tools)
        if !calls.isEmpty { return calls }
        if let pythonCall = parsePythonStyleToolFallback(in: buffer, tools: tools) {
            return [pythonCall]
        }
        if let bareNameJSONCall = parseBareNameJSONToolFallback(
            in: buffer,
            tools: tools,
            allowMalformedEOF: true)
        {
            return [bareNameJSONCall]
        }
        if let bareNameKeyValueCall = parseBareNameKeyValueToolFallback(
            in: buffer,
            tools: tools,
            allowEOF: true)
        {
            return [bareNameKeyValueCall]
        }
        if let actionJSONCall = parseActionJSONToolFallback(in: buffer, tools: tools) {
            return [actionJSONCall]
        }
        return parseInlineJSONToolFallback(in: buffer, tools: tools).map { [$0] } ?? []
    }

    // MARK: - Internals

    /// Parse the first `<｜DSML｜invoke name="...">...</｜DSML｜invoke>`
    /// in `text`. Returns nil when no well-formed invoke is found.
    private func parseFirstInvoke(
        in text: String, tools: [[String: any Sendable]]?
    ) -> ToolCall? {
        let invokes = parseAllInvokes(in: text, tools: tools)
        return invokes.first
    }

    /// Enumerate every `<｜DSML｜invoke name="NAME">PARAMS</｜DSML｜invoke>`
    /// in order. Robust to whitespace, multi-line formatting, and
    /// interleaved text. Parameters with `string="true"` are kept
    /// as raw strings; `string="false"` are JSON-parsed and
    /// converted through `JSONValue` so the emitted arguments match
    /// the tool schema's type contract.
    private func parseAllInvokes(
        in text: String, tools: [[String: any Sendable]]?
    ) -> [ToolCall] {
        let invokeOpen = "\(Self.dsmlPrefix)invoke name="
        let invokeCloseTags = [
            "\(Self.dsmlPrefixClose)invoke>",
            // Live DSV4-Flash sometimes abbreviates the closing invoke tag
            // while keeping the outer DSML envelope and parameters valid.
            "\(Self.dsmlPrefixClose)inv>",
        ]

        var results: [ToolCall] = []
        var cursor = text.startIndex
        while let open = text.range(of: invokeOpen, range: cursor ..< text.endIndex) {
            // Extract name: `"NAME">`
            let afterOpenEqual = open.upperBound
            guard
                let closeAngle = text.range(
                    of: ">", range: afterOpenEqual ..< text.endIndex)
            else { break }
            let headerRaw = text[afterOpenEqual ..< closeAngle.lowerBound]
                .trimmingCharacters(in: .whitespaces)
            // Header is like `"get_weather"`
            let funcName = stripQuotes(headerRaw)
            guard !funcName.isEmpty else {
                cursor = closeAngle.upperBound
                continue
            }

            // Find </｜DSML｜invoke>
            guard let close = firstRange(
                of: invokeCloseTags, in: text, range: closeAngle.upperBound ..< text.endIndex)
            else { break }

            let body = String(text[closeAngle.upperBound ..< close.lowerBound])
            let paramConfig = parameterSchema(for: funcName, tools: tools)
            let args = parseParameters(in: body, schema: paramConfig)
            results.append(
                ToolCall(function: .init(name: funcName, arguments: args)))
            cursor = close.upperBound
        }
        return results
    }

    private func firstRange(
        of needles: [String], in text: String, range: Range<String.Index>
    ) -> Range<String.Index>? {
        needles
            .compactMap { text.range(of: $0, range: range) }
            .min { $0.lowerBound < $1.lowerBound }
    }

    private func strippedOuterToolTags(from content: String) -> String {
        var text = content
        for start in startTagAliases {
            text = text.replacingOccurrences(of: start, with: "")
        }
        for end in endTagAliases {
            text = text.replacingOccurrences(of: end, with: "")
        }
        text = removingCompletedTags(withPrefix: Self.dsmlToolStartPrefix, from: text)
        text = removingCompletedTags(withPrefix: Self.dsmlToolEndPrefix, from: text)
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func removingCompletedTags(withPrefix prefix: String, from content: String) -> String {
        var text = content
        var searchStart = text.startIndex
        while
            let prefixRange = text.range(of: prefix, range: searchStart ..< text.endIndex),
            let close = text[prefixRange.upperBound...].firstIndex(of: ">")
        {
            let removalEnd = text.index(after: close)
            text.removeSubrange(prefixRange.lowerBound ..< removalEnd)
            searchStart = prefixRange.lowerBound
        }
        return text
    }

    /// Enumerate every `<｜DSML｜parameter name="NAME" string="BOOL">VALUE</｜DSML｜parameter>`
    /// in the invoke body. Values are decoded by the `string=` flag:
    ///   string="true"  → plain string preserved verbatim
    ///   string="false" → JSON-decoded (falls back to raw string if
    ///                    JSON parse fails, so a malformed model
    ///                    output doesn't drop the whole tool call).
    private func parseParameters(
        in body: String, schema: [String: any Sendable]?
    ) -> [String: any Sendable] {
        let paramOpen = "\(Self.dsmlPrefix)parameter name="
        let paramClose = "\(Self.dsmlPrefixClose)parameter>"

        var args: [String: any Sendable] = [:]
        var cursor = body.startIndex
        while let open = body.range(of: paramOpen, range: cursor ..< body.endIndex) {
            // The header is `"NAME" string="BOOL">`. Find closing `>`.
            let afterOpenEqual = open.upperBound
            guard
                let closeAngle = body.range(
                    of: ">", range: afterOpenEqual ..< body.endIndex)
            else { break }
            let headerRaw = String(body[afterOpenEqual ..< closeAngle.lowerBound])
            // Find the name quote.
            let name = extractAttrValue("", from: headerRaw, firstAnonymous: true)
            let stringFlag = extractAttrValue("string", from: headerRaw)

            guard !name.isEmpty else {
                cursor = closeAngle.upperBound
                continue
            }

            // Body up to </｜DSML｜parameter>
            guard
                let paramEnd = body.range(
                    of: paramClose, range: closeAngle.upperBound ..< body.endIndex)
            else { break }

            var value = String(body[closeAngle.upperBound ..< paramEnd.lowerBound])
            // Strip single leading / trailing newline (matches the
            // Python reference in `encoding_dsv4.py` which injects
            // newlines around multi-line values for readability).
            if value.hasPrefix("\n") { value = String(value.dropFirst()) }
            if value.hasSuffix("\n") { value = String(value.dropLast()) }

            if stringFlag == "true" {
                args[name] = value
            } else {
                // string="false" (or missing) → JSON decode.
                args[name] = decodeJSONValue(value, fallbackString: value)
            }
            _ = schema  // schema currently unused — DSML carries explicit `string=` flag so type hints aren't needed
            cursor = paramEnd.upperBound
        }
        return args
    }

    /// DSV4 live rows have occasionally fallen back from DSML to a top-level
    /// JSON object such as `{"tool":"file_read", ...}`. Treat that as a tool
    /// intent only when the named tool is present in the provided schema list.
    /// Unknown names and ordinary JSON answers remain visible text.
    private func parseInlineJSONToolFallback(
        in text: String,
        tools: [[String: any Sendable]]?
    ) -> ToolCall? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("{"), trimmed.hasSuffix("}"),
            let data = trimmed.data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }

        // Tool result envelopes can also carry a `"tool"` field. Those
        // are not new invocations, so leave them to the visible/text path.
        let hasSchemaList = tools?.isEmpty == false
        return parseInlineJSONObject(object, tools: tools, allowBareName: hasSchemaList)
    }

    /// Live DSV4 JANGTQ2 app rows can emit an agent-style action envelope,
    /// for example `action:{"id":0,"name":"file_read","args":{...}}`.
    /// Treat that as protocol, not assistant prose. If the action JSON is
    /// malformed but still names a registered tool, quarantine it as an
    /// invalid tool attempt so the visible stream stays clean.
    private func parseActionJSONToolFallback(
        in text: String,
        tools: [[String: any Sendable]]?
    ) -> ToolCall? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let payload = actionJSONPayload(in: trimmed) else { return nil }
        let hasSchemaList = tools?.isEmpty == false
        if let object = parseJSONObjectAllowingLiteralInvalidEscapes(payload) {
            return parseInlineJSONObject(object, tools: tools, allowBareName: hasSchemaList)
        }
        guard let name = actionJSONToolName(in: payload, allowBareName: hasSchemaList)
        else { return nil }
        if let tools, !tools.isEmpty {
            guard functionSpec(named: name, in: tools) != nil else { return nil }
        } else if !Self.schemaLessFallbackToolNames.contains(name) {
            return nil
        }
        return ToolCall(
            function: .init(
                name: name,
                arguments: invalidInlineToolArguments(
                    toolName: name,
                    message: "malformed JSON action object",
                    field: "arguments",
                    expected: "valid JSON object")))
    }

    private func parseInlineJSONObject(
        _ object: [String: Any],
        tools: [[String: any Sendable]]?,
        allowBareName: Bool
    ) -> ToolCall? {
        // Tool result envelopes can also carry a `"tool"` field. Those
        // are not new invocations, so leave them to the visible/text path.
        guard object["ok"] == nil, object["result"] == nil else { return nil }
        guard let name = fallbackToolName(in: object, allowBareName: allowBareName)
        else { return nil }
        let argsObject: Any
        if let function = object["function"] as? [String: Any] {
            if let arguments = function["arguments"] {
                argsObject = arguments
            } else if let parameters = function["parameters"] {
                argsObject = parameters
            } else {
                argsObject = [:] as [String: Any]
            }
        } else if let arguments = object["arguments"] {
            argsObject = arguments
        } else if let parameters = object["parameters"] {
            argsObject = parameters
        } else if let args = object["args"] {
            argsObject = args
        } else {
            let reserved = Set([
                "tool", "tool_name", "name", "function", "arguments", "parameters", "args",
                "type", "id",
            ])
            argsObject = object.filter { !reserved.contains($0.key) }
        }

        let args = normalizeInlineArguments(argsObject)
        if let tools, !tools.isEmpty {
            guard let spec = functionSpec(named: name, in: tools) else { return nil }
            if let invalidArgs = inlineSchemaValidationFailure(
                toolName: name,
                arguments: args,
                functionSpec: spec)
            {
                return ToolCall(function: .init(name: name, arguments: invalidArgs))
            }
        }
        return ToolCall(function: .init(name: name, arguments: args))
    }

    private func actionJSONPayload(in text: String) -> String? {
        let prefixes = ["action:", ":"]
        for prefix in prefixes where text.hasPrefix(prefix) {
            guard
                let prefixEnd = text.index(
                    text.startIndex,
                    offsetBy: prefix.count,
                    limitedBy: text.endIndex)
            else { return nil }
            var cursor = prefixEnd
            while cursor < text.endIndex, isInlineFallbackWhitespace(text[cursor]) {
                cursor = text.index(after: cursor)
            }
            if text[cursor...].lowercased().hasPrefix("json") {
                guard let afterJSON = text.index(cursor, offsetBy: 4, limitedBy: text.endIndex)
                else { return nil }
                cursor = afterJSON
                while cursor < text.endIndex, isInlineFallbackWhitespace(text[cursor]) {
                    cursor = text.index(after: cursor)
                }
            }
            guard cursor < text.endIndex, text[cursor] == "{" else { return nil }
            return firstBalancedJSONObject(in: text[cursor...]) ?? String(text[cursor...])
        }
        return nil
    }

    private func actionJSONToolName(in text: String, allowBareName: Bool) -> String? {
        for key in allowBareName ? ["tool", "tool_name", "name"] : ["tool", "tool_name"] {
            if let value = firstJSONStringValue(for: key, in: text) {
                return value
            }
        }
        if let functionRange = text.range(of: #""function""#),
            let value = firstJSONStringValue(for: "name", in: String(text[functionRange.upperBound...]))
        {
            return value
        }
        return nil
    }

    private func firstJSONStringValue(for key: String, in text: String) -> String? {
        let pattern = #""# + NSRegularExpression.escapedPattern(for: key)
            + #""\s*:\s*"((?:[^"\\]|\\.)*)""#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let range = NSRange(text.startIndex ..< text.endIndex, in: text)
        guard let match = regex.firstMatch(in: text, range: range),
            let valueRange = Range(match.range(at: 1), in: text)
        else { return nil }
        let raw = String(text[valueRange])
        let literal = "\"\(raw)\""
        guard let data = literal.data(using: .utf8),
            let decoded = try? JSONDecoder().decode(String.self, from: data)
        else { return raw }
        return decoded
    }

    /// Some live DSV4 JANGTQ2 app rows try to invoke folder tools as a
    /// Python-like function with JSON-style labels, for example:
    /// `file_read("path": "...", "start_line": 33, "end_line": 39)`.
    /// Treat that as a tool attempt instead of visible answer text.
    private func parsePythonStyleToolFallback(
        in text: String,
        tools: [[String: any Sendable]]?
    ) -> ToolCall? {
        guard
            let candidate = firstPythonStyleToolCallCandidate(in: text, tools: tools)
        else { return nil }

        let args = normalizePythonStyleArguments(candidate.arguments)
        if let tools, !tools.isEmpty {
            guard let spec = functionSpec(named: candidate.name, in: tools) else { return nil }
            if let invalidArgs = inlineSchemaValidationFailure(
                toolName: candidate.name,
                arguments: args,
                functionSpec: spec)
            {
                return ToolCall(function: .init(name: candidate.name, arguments: invalidArgs))
            }
        } else if !Self.schemaLessFallbackToolNames.contains(candidate.name) {
            return nil
        }
        return ToolCall(function: .init(name: candidate.name, arguments: args))
    }

    /// Live DSV4 JANGTQ2 app rows have also emitted a bare tool name followed
    /// by a JSON argument object, for example:
    ///
    ///     file_read
    ///     {"path":"/tmp/file.swift","start_line":1}
    ///
    /// Treat that as a tool attempt instead of visible prose. The JSON object is
    /// parsed as the argument payload for the preceding known tool name; trailing
    /// text is ignored so the streaming processor can stop before post-tool prose
    /// leaks into the answer.
    private func parseBareNameJSONToolFallback(
        in text: String,
        tools: [[String: any Sendable]]?,
        allowMalformedEOF: Bool
    ) -> ToolCall? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !trimmed.hasPrefix("{") else { return nil }

        for name in pythonStyleToolNames(from: tools) {
            guard trimmed.hasPrefix(name) else { continue }
            guard
                let afterName = trimmed.index(
                    trimmed.startIndex,
                    offsetBy: name.count,
                    limitedBy: trimmed.endIndex)
            else { continue }
            guard afterName == trimmed.endIndex
                || isInlineFallbackWhitespace(trimmed[afterName])
                || trimmed[afterName] == "{"
                || trimmed[afterName] == ":"
            else { continue }

            let cursor = bareNameJSONTailObjectStart(in: trimmed, afterName: afterName)
            guard cursor < trimmed.endIndex, trimmed[cursor] == "{" else { continue }
            guard let jsonObject = firstBalancedJSONObject(in: trimmed[cursor...]) else {
                if (allowMalformedEOF || jsonBracesBalanced(String(trimmed[cursor...]))),
                    let tools, !tools.isEmpty,
                    functionSpec(named: name, in: tools) != nil
                {
                    let invalidArgs = invalidInlineToolArguments(
                        toolName: name,
                        message: "malformed JSON argument object",
                        field: "arguments",
                        expected: "valid JSON object")
                    return ToolCall(function: .init(name: name, arguments: invalidArgs))
                }
                continue
            }
            guard let object = parseJSONObjectAllowingLiteralInvalidEscapes(jsonObject) else {
                if let tools, !tools.isEmpty {
                    guard functionSpec(named: name, in: tools) != nil else { return nil }
                    let invalidArgs = invalidInlineToolArguments(
                        toolName: name,
                        message: "malformed JSON argument object",
                        field: "arguments",
                        expected: "valid JSON object")
                    return ToolCall(function: .init(name: name, arguments: invalidArgs))
                }
                return nil
            }

            let args = normalizeInlineArguments(object)
            if let tools, !tools.isEmpty {
                guard let spec = functionSpec(named: name, in: tools) else { return nil }
                if let invalidArgs = inlineSchemaValidationFailure(
                    toolName: name,
                    arguments: args,
                    functionSpec: spec)
                {
                    return ToolCall(function: .init(name: name, arguments: invalidArgs))
                }
            } else if !Self.schemaLessFallbackToolNames.contains(name) {
                return nil
            }
            return ToolCall(function: .init(name: name, arguments: args))
        }
        return nil
    }

    private func jsonBracesBalanced(_ text: String) -> Bool {
        var depth = 0
        var sawOpen = false
        for ch in text {
            if ch == "{" {
                sawOpen = true
                depth += 1
            } else if ch == "}" {
                depth -= 1
                if depth < 0 { return false }
            }
        }
        return sawOpen && depth == 0
    }

    private func parseJSONObjectAllowingLiteralInvalidEscapes(_ jsonObject: String) -> [String: Any]? {
        if let object = parseJSONObject(jsonObject) {
            return object
        }
        let repaired = jsonObjectWithInvalidStringEscapesMadeLiteral(jsonObject)
        guard repaired != jsonObject else { return nil }
        return parseJSONObject(repaired)
    }

    private func parseJSONObject(_ jsonObject: String) -> [String: Any]? {
        guard let data = jsonObject.data(using: .utf8) else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    private func jsonObjectWithInvalidStringEscapesMadeLiteral(_ jsonObject: String) -> String {
        var output = ""
        var inString = false
        var escaped = false
        let validEscapes = Set(["\"", "\\", "/", "b", "f", "n", "r", "t", "u"])

        for ch in jsonObject {
            if escaped {
                let scalar = String(ch)
                if !validEscapes.contains(scalar) {
                    output.append("\\")
                }
                output.append(ch)
                escaped = false
                continue
            }
            if inString && ch == "\\" {
                output.append(ch)
                escaped = true
                continue
            }
            if ch == "\"" {
                inString.toggle()
            }
            output.append(ch)
        }
        if escaped {
            output.append("\\")
        }
        return output
    }

    /// Live DSV4 JANGTQ2 UI rows can emit a bare tool name followed by
    /// shell-like key/value arguments, for example:
    ///
    ///     file_read
    ///     path=/tmp/file.swift
    ///
    /// This is a tool attempt, not assistant prose. Parse only consecutive
    /// key/value lines immediately after a known tool name; trailing text is
    /// deliberately ignored so a post-tool answer cannot leak before the tool
    /// result has been fed back through the chat loop.
    private func parseBareNameKeyValueToolFallback(
        in text: String,
        tools: [[String: any Sendable]]?,
        allowEOF: Bool
    ) -> ToolCall? {
        var leadingStart = text.startIndex
        while leadingStart < text.endIndex, isInlineFallbackWhitespace(text[leadingStart]) {
            leadingStart = text.index(after: leadingStart)
        }
        let trimmed = String(text[leadingStart...])
        guard !trimmed.isEmpty, !trimmed.hasPrefix("{") else { return nil }

        for name in pythonStyleToolNames(from: tools) {
            guard trimmed.hasPrefix(name) else { continue }
            guard
                let afterName = trimmed.index(
                    trimmed.startIndex,
                    offsetBy: name.count,
                    limitedBy: trimmed.endIndex)
            else { continue }
            guard afterName < trimmed.endIndex,
                isInlineFallbackWhitespace(trimmed[afterName])
            else { continue }

            let tail = String(trimmed[afterName...])
            let args = parseLeadingKeyValueLines(tail)
            guard !args.isEmpty else { continue }
            guard allowEOF || bareNameKeyValueTailIsTerminated(tail, toolName: name, tools: tools)
            else { continue }

            if let tools, !tools.isEmpty {
                guard let spec = functionSpec(named: name, in: tools) else { return nil }
                if let invalidArgs = inlineSchemaValidationFailure(
                    toolName: name,
                    arguments: args,
                    functionSpec: spec)
                {
                    return ToolCall(function: .init(name: name, arguments: invalidArgs))
                }
            } else if !Self.schemaLessFallbackToolNames.contains(name) {
                return nil
            }
            return ToolCall(function: .init(name: name, arguments: args))
        }
        return nil
    }

    private func parseLeadingKeyValueLines(_ text: String) -> [String: any Sendable] {
        var result: [String: any Sendable] = [:]
        for rawLine in text.split(whereSeparator: \.isNewline) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }
            guard let separator = firstKeyValueSeparator(in: line) else {
                break
            }
            let key = line[..<separator]
                .trimmingCharacters(in: .whitespaces)
            guard !key.isEmpty, key.allSatisfy(isKeyValueIdentifierCharacter) else {
                break
            }
            var value = line[line.index(after: separator)...]
                .trimmingCharacters(in: .whitespaces)
            guard !value.isEmpty else { break }
            if (value.hasPrefix("'") && value.hasSuffix("'"))
                || (value.hasPrefix("\"") && value.hasSuffix("\""))
            {
                value = String(value.dropFirst().dropLast())
                result[key] = value
            } else {
                result[key] = decodeJSONValue(value, fallbackString: value)
            }
        }
        return result
    }

    private func bareNameKeyValueTailIsTerminated(
        _ text: String,
        toolName: String,
        tools: [[String: any Sendable]]?
    ) -> Bool {
        let lines = text.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline)
        let endsWithNewline = text.last?.isNewline == true
        var sawKeyValue = false
        for (index, rawLine) in lines.enumerated() {
            let isLastLine = index == lines.count - 1
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty {
                if isLastLine { return false }
                if sawKeyValue { return true }
                continue
            }
            if isKeyValueArgumentLine(line, toolName: toolName, tools: tools) {
                sawKeyValue = true
                continue
            }
            if sawKeyValue {
                if isLastLine && !endsWithNewline
                    && (line.allSatisfy(isKeyValueIdentifierCharacter)
                        || isPartialKeyValueArgumentLine(line, toolName: toolName, tools: tools))
                {
                    return false
                }
                return true
            }
            return false
        }
        return false
    }

    private func isPartialKeyValueArgumentLine(
        _ line: String,
        toolName: String,
        tools: [[String: any Sendable]]?
    ) -> Bool {
        guard let separator = firstKeyValueSeparator(in: line) else { return false }
        let key = line[..<separator]
            .trimmingCharacters(in: .whitespaces)
        guard !key.isEmpty, key.allSatisfy(isKeyValueIdentifierCharacter) else {
            return false
        }
        let value = line[line.index(after: separator)...]
            .trimmingCharacters(in: .whitespaces)
        guard value.isEmpty else { return false }
        if let allowed = keyValueArgumentNames(for: toolName, tools: tools),
            allowed.contains(String(key))
        {
            return true
        }
        if Self.schemaLessKeyValueArgumentNames.contains(String(key)) {
            return true
        }
        if let allowed = keyValueArgumentNames(for: toolName, tools: tools), !allowed.isEmpty {
            return allowed.contains(String(key))
        }
        return false
    }

    private func isKeyValueArgumentLine(
        _ line: String,
        toolName: String,
        tools: [[String: any Sendable]]?
    ) -> Bool {
        guard let separator = firstKeyValueSeparator(in: line) else { return false }
        let key = line[..<separator]
            .trimmingCharacters(in: .whitespaces)
        guard !key.isEmpty, key.allSatisfy(isKeyValueIdentifierCharacter) else {
            return false
        }
        let value = line[line.index(after: separator)...]
            .trimmingCharacters(in: .whitespaces)
        guard !value.isEmpty else { return false }
        if let allowed = keyValueArgumentNames(for: toolName, tools: tools),
            allowed.contains(String(key))
        {
            return true
        }
        if Self.schemaLessKeyValueArgumentNames.contains(String(key)) {
            return true
        }
        if let allowed = keyValueArgumentNames(for: toolName, tools: tools), !allowed.isEmpty {
            return allowed.contains(String(key))
        }
        return false
    }

    private func keyValueArgumentNames(
        for toolName: String,
        tools: [[String: any Sendable]]?
    ) -> Set<String>? {
        guard let tools else { return nil }
        for tool in tools {
            let function = (tool["function"] as? [String: any Sendable]) ?? tool
            guard function["name"] as? String == toolName else { continue }
            guard
                let parameters = function["parameters"] as? [String: any Sendable],
                let properties = parameters["properties"] as? [String: any Sendable]
            else { return nil }
            return Set(properties.keys)
        }
        return nil
    }

    private func firstKeyValueSeparator(in line: String) -> String.Index? {
        let equals = line.firstIndex(of: "=")
        let colon = line.firstIndex(of: ":")
        switch (equals, colon) {
        case (let e?, let c?):
            return e < c ? e : c
        case (let e?, nil):
            return e
        case (nil, let c?):
            return c
        case (nil, nil):
            return nil
        }
    }

    private func isKeyValueIdentifierCharacter(_ character: Character) -> Bool {
        character == "_" || character.isLetter || character.isNumber
    }

    private func bareNameJSONTailObjectStart(
        in text: String,
        afterName: String.Index
    ) -> String.Index {
        var cursor = afterName
        while cursor < text.endIndex, isInlineFallbackWhitespace(text[cursor]) {
            cursor = text.index(after: cursor)
        }
        cursor = consumeOptionalJSONLabel(in: text, at: cursor)
        guard cursor < text.endIndex else { return cursor }
        guard text[cursor...].hasPrefix("```") else { return cursor }

        var fenceCursor = cursor
        for _ in 0..<3 {
            guard fenceCursor < text.endIndex else { return cursor }
            fenceCursor = text.index(after: fenceCursor)
        }
        while fenceCursor < text.endIndex,
            text[fenceCursor] != "\n",
            text[fenceCursor] != "\r"
        {
            fenceCursor = text.index(after: fenceCursor)
        }
        while fenceCursor < text.endIndex, isInlineFallbackWhitespace(text[fenceCursor]) {
            fenceCursor = text.index(after: fenceCursor)
        }
        return fenceCursor
    }

    private func consumeOptionalJSONLabel(in text: String, at cursor: String.Index) -> String.Index {
        guard cursor < text.endIndex, text[cursor] == ":" else { return cursor }
        var labelStart = text.index(after: cursor)
        while labelStart < text.endIndex, isInlineFallbackWhitespace(text[labelStart]) {
            labelStart = text.index(after: labelStart)
        }
        guard text[labelStart...].lowercased().hasPrefix("json") else { return cursor }
        guard
            let labelEnd = text.index(labelStart, offsetBy: 4, limitedBy: text.endIndex)
        else { return cursor }
        if labelEnd < text.endIndex,
            !isInlineFallbackWhitespace(text[labelEnd]),
            text[labelEnd] != "{",
            !text[labelEnd...].hasPrefix("```")
        {
            return cursor
        }
        var next = labelEnd
        while next < text.endIndex, isInlineFallbackWhitespace(text[next]) {
            next = text.index(after: next)
        }
        return next
    }

    private static let schemaLessFallbackToolNames: Set<String> = [
        "file_tree",
        "file_read",
        "file_write",
        "file_edit",
        "file_search",
        "shell_run",
        "git_status",
        "git_diff",
        "git_commit",
    ]

    private static let schemaLessKeyValueArgumentNames: Set<String> = [
        "path",
        "start_line",
        "end_line",
        "content",
        "command",
        "query",
        "pattern",
        "replacement",
        "old",
        "new",
        "message",
        "cwd",
        "recursive",
    ]

    private func firstPythonStyleToolCallCandidate(
        in text: String,
        tools: [[String: any Sendable]]?
    ) -> (name: String, arguments: String)? {
        let names = pythonStyleToolNames(from: tools)
        for name in names {
            var searchStart = text.startIndex
            let needle = "\(name)("
            while let range = text.range(of: needle, range: searchStart ..< text.endIndex) {
                guard isFunctionNameBoundary(text, before: range.lowerBound) else {
                    searchStart = range.upperBound
                    continue
                }
                let openParen = text.index(before: range.upperBound)
                guard let closeParen = matchingCloseParen(in: text, openParen: openParen) else {
                    return nil
                }
                let argsStart = text.index(after: openParen)
                return (name, String(text[argsStart ..< closeParen]))
            }
        }
        return nil
    }

    private func pythonStyleToolNames(from tools: [[String: any Sendable]]?) -> [String] {
        let schemaNames = tools?.compactMap { tool -> String? in
            let function = (tool["function"] as? [String: any Sendable]) ?? tool
            return function["name"] as? String
        } ?? []
        if !schemaNames.isEmpty {
            return schemaNames.sorted { $0.count > $1.count }
        }
        return Self.schemaLessFallbackToolNames.sorted { $0.count > $1.count }
    }

    private func isFunctionNameBoundary(_ text: String, before index: String.Index) -> Bool {
        guard index > text.startIndex else { return true }
        let previous = text[text.index(before: index)]
        return !(previous.isLetter || previous.isNumber || previous == "_")
    }

    private func matchingCloseParen(in text: String, openParen: String.Index) -> String.Index? {
        var depth = 0
        var quote: Character?
        var escaped = false
        var cursor = openParen
        while cursor < text.endIndex {
            let ch = text[cursor]
            defer { cursor = text.index(after: cursor) }
            if escaped {
                escaped = false
                continue
            }
            if ch == "\\" {
                escaped = quote != nil
                continue
            }
            if let activeQuote = quote {
                if ch == activeQuote { quote = nil }
                continue
            }
            if ch == "\"" || ch == "'" {
                quote = ch
                continue
            }
            if ch == "(" {
                depth += 1
            } else if ch == ")" {
                depth -= 1
                if depth == 0 { return cursor }
                if depth < 0 { return nil }
            }
        }
        return nil
    }

    private func firstBalancedJSONObject(in text: Substring) -> String? {
        guard let open = text.firstIndex(of: "{") else { return nil }
        var depth = 0
        var quote: Character?
        var escaped = false
        var cursor = open
        while cursor < text.endIndex {
            let ch = text[cursor]
            defer { cursor = text.index(after: cursor) }
            if escaped {
                escaped = false
                continue
            }
            if ch == "\\" {
                escaped = quote != nil
                continue
            }
            if let activeQuote = quote {
                if ch == activeQuote { quote = nil }
                continue
            }
            if ch == "\"" {
                quote = ch
                continue
            }
            if ch == "{" {
                depth += 1
            } else if ch == "}" {
                depth -= 1
                if depth == 0 {
                    return String(text[open...cursor])
                }
                if depth < 0 { return nil }
            }
        }
        return nil
    }

    private func isInlineFallbackWhitespace(_ character: Character) -> Bool {
        character.unicodeScalars.allSatisfy {
            CharacterSet.whitespacesAndNewlines.contains($0)
        }
    }

    private func normalizePythonStyleArguments(_ arguments: String) -> [String: any Sendable] {
        let trimmed = arguments.trimmingCharacters(in: .whitespacesAndNewlines)
        if let data = "{\(trimmed)}".data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data),
            let normalized = normalizeJSON(object) as? [String: any Sendable]
        {
            return normalized
        }
        return normalizePythonKeywordArguments(trimmed)
    }

    private func normalizePythonKeywordArguments(_ arguments: String) -> [String: any Sendable] {
        var result: [String: any Sendable] = [:]
        let pattern = #"(\w+)\s*=\s*('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|[^,\)]+)"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return result }
        let range = NSRange(arguments.startIndex ..< arguments.endIndex, in: arguments)
        for match in regex.matches(in: arguments, range: range) {
            guard
                let keyRange = Range(match.range(at: 1), in: arguments),
                let valueRange = Range(match.range(at: 2), in: arguments)
            else { continue }
            let key = String(arguments[keyRange])
            var value = String(arguments[valueRange]).trimmingCharacters(in: .whitespaces)
            if (value.hasPrefix("'") && value.hasSuffix("'"))
                || (value.hasPrefix("\"") && value.hasSuffix("\""))
            {
                value = String(value.dropFirst().dropLast())
                result[key] = value
            } else {
                result[key] = decodeJSONValue(value, fallbackString: value)
            }
        }
        return result
    }

    private func fallbackToolName(in object: [String: Any], allowBareName: Bool) -> String? {
        if let tool = object["tool"] as? String { return tool }
        if let toolName = object["tool_name"] as? String { return toolName }
        if let function = object["function"] as? [String: Any],
            let name = function["name"] as? String
        {
            return name
        }
        if allowBareName, let name = object["name"] as? String { return name }
        return nil
    }

    private func functionSpec(
        named name: String,
        in tools: [[String: any Sendable]]
    ) -> [String: any Sendable]? {
        for tool in tools {
            let function = (tool["function"] as? [String: any Sendable]) ?? tool
            if function["name"] as? String == name {
                return function
            }
        }
        return nil
    }

    private func inlineSchemaValidationFailure(
        toolName: String,
        arguments: [String: any Sendable],
        functionSpec: [String: any Sendable]
    ) -> [String: any Sendable]? {
        guard let parameters = sendableObject(functionSpec["parameters"]) else {
            return nil
        }

        for required in sendableStringArray(parameters["required"]) {
            if arguments[required] == nil {
                return invalidInlineToolArguments(
                    toolName: toolName,
                    message: "missing required argument: \(required)",
                    field: required,
                    expected: "required parameter")
            }
        }

        guard sendableBool(parameters["additionalProperties"]) == false,
            let properties = sendableObject(parameters["properties"])
        else { return nil }

        let allowed = Set(properties.keys)
        if let unknown = arguments.keys.sorted().first(where: { !allowed.contains($0) }) {
            return invalidInlineToolArguments(
                toolName: toolName,
                message: "unknown argument: \(unknown)",
                field: unknown,
                expected: "declared parameter")
        }
        return nil
    }

    private func invalidInlineToolArguments(
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
        if case .object(let object)? = value as? JSONValue {
            return object.mapValues { $0.sendableValue }
        }
        return nil
    }

    private func sendableStringArray(_ value: (any Sendable)?) -> [String] {
        if let strings = value as? [String] {
            return strings
        }
        if let values = value as? [any Sendable] {
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

    private func sendableBool(_ value: (any Sendable)?) -> Bool? {
        if let bool = value as? Bool {
            return bool
        }
        if let number = value as? NSNumber {
            return number.boolValue
        }
        if case .bool(let bool)? = value as? JSONValue {
            return bool
        }
        return nil
    }

    private func normalizeInlineArguments(_ value: Any) -> [String: any Sendable] {
        if let string = value as? String,
            let data = string.data(using: .utf8),
            let parsed = try? JSONSerialization.jsonObject(with: data),
            let normalized = normalizeJSON(parsed) as? [String: any Sendable]
        {
            return normalized
        }
        if let normalized = normalizeJSON(value) as? [String: any Sendable] {
            return normalized
        }
        return [:]
    }

    /// Extract `attr="VALUE"` from a header string. When
    /// `firstAnonymous` is true and attr is empty, the first quoted
    /// literal in the header is returned (used to pull the invoke /
    /// parameter `name` which is the first bare `"..."`).
    private func extractAttrValue(
        _ attr: String, from header: String, firstAnonymous: Bool = false
    ) -> String {
        if firstAnonymous {
            // First double-quoted literal in header.
            guard let first = header.firstIndex(of: "\"") else { return "" }
            let after = header.index(after: first)
            guard let second = header[after...].firstIndex(of: "\"") else { return "" }
            return String(header[after ..< second])
        }
        let needle = "\(attr)="
        guard let r = header.range(of: needle) else { return "" }
        let afterEq = r.upperBound
        guard afterEq < header.endIndex, header[afterEq] == "\"" else { return "" }
        let valueStart = header.index(after: afterEq)
        guard let valueEnd = header[valueStart...].firstIndex(of: "\"") else { return "" }
        return String(header[valueStart ..< valueEnd])
    }

    private func stripQuotes(_ s: String) -> String {
        var t = s
        if t.hasPrefix("\"") { t = String(t.dropFirst()) }
        if t.hasSuffix("\"") { t = String(t.dropLast()) }
        return t
    }

    /// Best-effort JSON decode for `string="false"` parameter
    /// values. Falls back to the raw string when the parse fails —
    /// model outputs with small syntactic slip (trailing comma,
    /// smart quotes inside the value) shouldn't drop the whole tool
    /// call. We record the raw string in that case so the
    /// downstream consumer can still act on it.
    private func decodeJSONValue(_ raw: String, fallbackString: String) -> any Sendable {
        guard let data = raw.data(using: .utf8) else { return fallbackString }
        if let parsed = try? JSONSerialization.jsonObject(
            with: data, options: [.fragmentsAllowed])
        {
            return normalizeJSON(parsed) ?? fallbackString
        }
        return fallbackString
    }

    /// JSONSerialization returns `Any` + NSNumber; convert to a
    /// Swift-native `Sendable` tree so the downstream `ToolCall`
    /// struct can carry it safely across actor boundaries.
    private func normalizeJSON(_ any: Any) -> (any Sendable)? {
        switch any {
        case let s as String: return s
        case let n as NSNumber:
            // NSNumber can be bool, int, or double.
            if CFGetTypeID(n) == CFBooleanGetTypeID() {
                return n.boolValue
            }
            let d = n.doubleValue
            if d.rounded() == d, abs(d) < 1e18 {
                return Int(d)
            }
            return d
        case let a as [Any]:
            let mapped = a.compactMap { normalizeJSON($0) }
            return mapped
        case let d as [String: Any]:
            var out: [String: any Sendable] = [:]
            for (k, v) in d {
                if let nv = normalizeJSON(v) {
                    out[k] = nv
                }
            }
            return out
        case is NSNull:
            return Optional<String>.none as any Sendable
        default:
            return nil
        }
    }

    private func parameterSchema(
        for funcName: String, tools: [[String: any Sendable]]?
    ) -> [String: any Sendable]? {
        guard let tools else { return nil }
        for tool in tools {
            let function = (tool["function"] as? [String: any Sendable]) ?? tool
            if let name = function["name"] as? String, name == funcName {
                if let params = function["parameters"] as? [String: any Sendable],
                    let properties = params["properties"] as? [String: any Sendable]
                {
                    return properties
                }
            }
        }
        return nil
    }
}
