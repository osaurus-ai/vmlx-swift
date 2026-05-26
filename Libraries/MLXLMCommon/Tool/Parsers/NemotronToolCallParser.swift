// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

/// Nemotron-H / Omni tool-call parser.
///
/// Nemotron templates in the wild advertise XML function calls, but live
/// JANGTQ Omni rows can emit DSML envelopes. This parser owns that family
/// boundary so XML-only families do not inherit DSML tolerance, while Nemotron
/// protocol text is still buffered and parsed as tool transport.
public struct NemotronToolCallParser: ToolCallParser, Sendable {
    private let xml = XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
    private let dsml = DSMLToolCallParser()

    public let startTag: String? = "<tool_call>"
    public let endTag: String? = "</tool_call>"

    public var startTagAliases: [String] {
        xml.startTagAliases + dsml.startTagAliases
    }

    public var endTagAliases: [String] {
        xml.endTagAliases + dsml.endTagAliases
    }

    public var startTagPrefixes: [String] {
        dsml.startTagPrefixes
    }

    public var endTagPrefixes: [String] {
        dsml.endTagPrefixes
    }

    public var supportsInlineJSONToolFallback: Bool {
        dsml.supportsInlineJSONToolFallback
    }

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        xml.parse(content: content, tools: tools)
            ?? dsml.parse(content: content, tools: tools)
    }

    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        let xmlCalls = xml.parseEOS(toolCallBuffer, tools: tools)
        if !xmlCalls.isEmpty { return xmlCalls }
        return dsml.parseEOS(toolCallBuffer, tools: tools)
    }

    public func isValidPartialContent(_ toolCallBuffer: String) -> Bool {
        xml.isValidPartialContent(toolCallBuffer) || dsml.isValidPartialContent(toolCallBuffer)
    }
}
