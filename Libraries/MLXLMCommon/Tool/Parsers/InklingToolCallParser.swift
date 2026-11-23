// Copyright © 2026 Jinho Jang. All rights reserved.

import Foundation

/// Inkling-2 (`inkling2`, `inkling2_loop`) tool calls.
///
/// The model writes `<|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{…}}<|end_message|>`
/// (one message per call, several allowed per turn). The Inkling reasoning parser consumes the
/// `<|message_model|>NAME` header and forwards `<|content_invoke_tool_json|>JSON<|end_message|>` on the
/// content channel, so this parser sees a plain tagged envelope. The JSON body is authoritative
/// (`name` + `args`); `arguments` / `parameters` are accepted as spellings of `args`.
public struct InklingToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<|content_invoke_tool_json|>"
    public let endTag: String? = "<|end_message|>"

    /// Inkling protocol markers that must never reach visible text if the reasoning parser did
    /// not run (e.g. a host that disabled reasoning parsing): stripped as orphans.
    public var orphanStripTags: [String] {
        [
            "<|message_model|>", "<|content_text|>", "<|content_thinking|>", "<|end_message|>",
            "<|content_model_end_sampling|>", "<|content_invoke_tool_text|>",
            "<|content_tool_error|>",
        ]
    }

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        var text = content
        for tag in [startTag!, endTag!, "<|content_model_end_sampling|>"] {
            text = text.replacingOccurrences(of: tag, with: "")
        }
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let object = tryParseJSON(text) as? [String: any Sendable],
            let name = object["name"] as? String, !name.isEmpty
        else { return nil }
        let rawArgs = object["args"] ?? object["arguments"] ?? object["parameters"]
        let arguments: [String: any Sendable]
        if let dict = rawArgs as? [String: any Sendable] {
            arguments = dict
        } else if let string = rawArgs as? String,
            let dict = tryParseJSON(string) as? [String: any Sendable]
        {
            arguments = dict
        } else if rawArgs == nil {
            arguments = [:]
        } else {
            return nil
        }
        return ToolCall(function: .init(name: name, arguments: arguments))
    }
}
