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

    public let startTag: String? = "<\u{FF5C}DSML\u{FF5C}tool_calls>"
    public let endTag: String? = "</\u{FF5C}DSML\u{FF5C}tool_calls>"
    public let startTagAliases: [String] = [
        "<\u{FF5C}DSML\u{FF5C}tool_calls>",
        // Live DeepSeek V4 Flash sometimes drops the second "l".
        "<\u{FF5C}DSML\u{FF5C}tool_cals>",
    ]
    public let endTagAliases: [String] = [
        "</\u{FF5C}DSML\u{FF5C}tool_calls>",
        "</\u{FF5C}DSML\u{FF5C}tool_cals>",
    ]

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        // Strip outer block if present.
        var text = content
        for start in startTagAliases {
            text = text.replacingOccurrences(of: start, with: "")
        }
        for end in endTagAliases {
            text = text.replacingOccurrences(of: end, with: "")
        }
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)

        // Find first <｜DSML｜invoke name="...">
        guard let firstCall = parseFirstInvoke(in: text, tools: tools) else {
            return nil
        }
        return firstCall
    }

    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall]
    {
        // DSML can carry multiple invokes within a single
        // <｜DSML｜tool_calls> block, so the default `components(by:
        // startTag)` strategy won't round-trip — each multi-invoke
        // block is ONE `startTag..endTag` outer envelope, split
        // internally by `<｜DSML｜invoke ...>` per call. Parse all
        // invokes in order.
        var buffer = toolCallBuffer
        for start in startTagAliases {
            buffer = buffer.replacingOccurrences(of: start, with: "")
        }
        for end in endTagAliases {
            buffer = buffer.replacingOccurrences(of: end, with: "")
        }
        return parseAllInvokes(in: buffer, tools: tools)
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
