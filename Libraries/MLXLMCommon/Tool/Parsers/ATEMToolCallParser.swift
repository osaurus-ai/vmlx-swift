// Copyright © 2026 Apple Inc.

import Foundation

/// Parser for the "Onyx ATEM" tool-call dialect used by Muse Glimmer.
///
/// ```
/// <atem:function_calls>
/// <atem:invoke name="get_weather">
/// <atem:parameter name="city">Paris</atem:parameter>
/// <atem:parameter name="days">3</atem:parameter>
/// </atem:invoke>
/// </atem:function_calls>
/// ```
///
/// Two properties of the wire format drive the implementation:
///
/// 1. **One envelope can carry several `<atem:invoke>` blocks.** `parse` returns
///    the first (the `ToolCallParser` contract is one call per parse) and
///    ``parseEOS`` returns all of them, so a parallel-call turn is not silently
///    truncated to its first function.
///
/// 2. **It is not real XML.** The model's own instructions say the output "is
///    not expected to be valid XML and is parsed with regular expressions", so
///    values may contain `<`, `>`, and unescaped quotes. Scanning is therefore
///    done by locating the *specific* `</atem:parameter>` closer rather than by
///    an XML parse or a `[^<]*` body match — a stray `<` inside a value used to
///    desync tag detection on gemma4 and leak the envelope as visible text.
public struct ATEMToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<atem:function_calls>"
    public let endTag: String? = "</atem:function_calls>"

    public init() {}

    /// The envelope's closers are protocol control markers even when orphaned.
    /// A model that emits a stray `</atem:invoke></atem:function_calls>` run
    /// after an agent-loop step must not surface it as assistant prose.
    public var orphanStripTags: [String] {
        ["</atem:function_calls>", "</atem:invoke>", "</atem:parameter>"]
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        parseAll(content: content, tools: tools).first
    }

    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?)
        -> [ToolCall]
    {
        parseAll(content: toolCallBuffer, tools: tools)
    }

    /// A buffer can still become a valid call while it is a prefix of the
    /// envelope, or once the envelope is open. Rejecting early matters: a
    /// literal `<atem:` in ordinary prose should surface as text rather than
    /// be buffered until end of generation.
    public func isValidPartialContent(_ toolCallBuffer: String) -> Bool {
        let trimmed = toolCallBuffer.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let open = startTag else { return true }
        if trimmed.hasPrefix(open) { return true }
        // Still typing the opener.
        return open.hasPrefix(trimmed)
    }

    // MARK: - Scanning

    public func parseAll(content: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        var calls: [ToolCall] = []
        var cursor = content.startIndex

        while let invokeOpen = content.range(of: "<atem:invoke", range: cursor ..< content.endIndex)
        {
            // The opener runs to the first `>` after the name attribute.
            guard
                let headerEnd = content.range(
                    of: ">", range: invokeOpen.upperBound ..< content.endIndex)
            else { break }

            let header = String(content[invokeOpen.upperBound ..< headerEnd.lowerBound])
            guard let name = Self.attribute("name", in: header), !name.isEmpty else {
                cursor = headerEnd.upperBound
                continue
            }

            // Body ends at this invoke's closer, or at the envelope closer, or
            // at end of buffer — a truncated final call still yields whatever
            // parameters completed.
            let bodyStart = headerEnd.upperBound
            let bodyEnd =
                content.range(of: "</atem:invoke>", range: bodyStart ..< content.endIndex)?
                .lowerBound
                ?? content.range(
                    of: "</atem:function_calls>", range: bodyStart ..< content.endIndex)?
                .lowerBound
                ?? content.endIndex

            let body = String(content[bodyStart ..< bodyEnd])
            let arguments = Self.parameters(in: body, funcName: name, tools: tools)
            calls.append(ToolCall(function: .init(name: name, arguments: arguments)))

            cursor = bodyEnd < content.endIndex ? content.index(after: bodyEnd) : content.endIndex
            if cursor >= content.endIndex { break }
        }

        return calls
    }

    /// Extract `<atem:parameter name="k">v</atem:parameter>` pairs.
    ///
    /// The value is everything up to the *next* `</atem:parameter>`, so values
    /// containing `<` or `>` survive intact.
    private static func parameters(
        in body: String, funcName: String, tools: [[String: any Sendable]]?
    ) -> [String: any Sendable] {
        var arguments: [String: any Sendable] = [:]
        var cursor = body.startIndex

        while let open = body.range(of: "<atem:parameter", range: cursor ..< body.endIndex) {
            guard
                let headerEnd = body.range(of: ">", range: open.upperBound ..< body.endIndex)
            else { break }

            let header = String(body[open.upperBound ..< headerEnd.lowerBound])
            let valueStart = headerEnd.upperBound
            let closer = body.range(of: "</atem:parameter>", range: valueStart ..< body.endIndex)
            let valueEnd = closer?.lowerBound ?? body.endIndex
            let rawValue = String(body[valueStart ..< valueEnd])

            if let key = Self.attribute("name", in: header), !key.isEmpty {
                arguments[key] = coerce(
                    rawValue, paramName: key, funcName: funcName, tools: tools)
            }

            cursor = closer?.upperBound ?? body.endIndex
        }

        return arguments
    }

    /// The template writes scalars bare and lists/objects/booleans as JSON.
    ///
    /// Schema-directed conversion runs first so a declared string parameter
    /// whose value happens to look like a number (a zip code, an ID) is not
    /// silently turned into an `Int`. Only when the schema says nothing do we
    /// fall back to shape-based JSON detection, and even then only for the
    /// unambiguous forms — a bare word stays a string.
    private static func coerce(
        _ raw: String, paramName: String, funcName: String, tools: [[String: any Sendable]]?
    ) -> any Sendable {
        // Leading/trailing newlines come from the template's own formatting,
        // not from the value.
        let value = raw.trimmingCharacters(in: .newlines)

        if getParameterType(funcName: funcName, paramName: paramName, tools: tools) != nil {
            return convertParameterValue(
                value, paramName: paramName, funcName: funcName, tools: tools)
        }

        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        switch trimmed {
        case "true": return true
        case "false": return false
        case "null": return NSNull()
        default: break
        }
        if let first = trimmed.first, first == "{" || first == "[",
            let json = tryParseJSON(trimmed)
        {
            return json
        }
        return value
    }

    /// Read `name="value"` (or `name='value'`) out of a tag header.
    ///
    /// Falls back to an unquoted run so `<atem:invoke name=get_weather>` — seen
    /// when a weak quant drops the quotes — still resolves instead of dropping
    /// the whole call.
    private static func attribute(_ attribute: String, in header: String) -> String? {
        guard let keyRange = header.range(of: "\(attribute)=") else { return nil }
        let rest = header[keyRange.upperBound...]
        guard let first = rest.first else { return nil }

        if first == "\"" || first == "'" {
            let afterQuote = rest.index(after: rest.startIndex)
            guard let closing = rest[afterQuote...].firstIndex(of: first) else { return nil }
            return String(rest[afterQuote ..< closing])
        }

        let unquoted = rest.prefix { !$0.isWhitespace && $0 != ">" && $0 != "/" }
        return unquoted.isEmpty ? nil : String(unquoted)
    }
}
