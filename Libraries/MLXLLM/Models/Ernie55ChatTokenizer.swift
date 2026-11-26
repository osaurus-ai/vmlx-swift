// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLXLMCommon

/// The released ERNIE tokenizer defines a native text protocol in place of a Jinja template.
struct Ernie55ChatTokenizer: GenerationPromptControllableTokenizer {
    let base: any Tokenizer

    private static let mask = "<|ERNIE_MASK|>"
    private static let start = "<|ernie:im_start|>"
    private static let end = "<|ernie:im_end|>"
    private static let think = "<ernie:think>"
    private static let thinkEnd = "</ernie:think>"
    private static let toolCall = "<ernie:tool_call>"
    private static let toolCallEnd = "</ernie:tool_call>"
    private static let toolOutput = "<ernie:tool_output>"
    private static let toolOutputEnd = "</ernie:tool_output>"
    private static let system = "<ernie:system_setting>"
    private static let systemEnd = "</ernie:system_setting>"
    private static let settings = "<ernie:global_setting>"
    private static let settingsEnd = "</ernie:global_setting>"
    private static let toolList = "<ernie:tool_list>"
    private static let toolListEnd = "</ernie:tool_list>"

    var grammarTokenVocabulary: GrammarTokenVocabulary? { base.grammarTokenVocabulary }
    var incrementalByteLevelDecoder: (@Sendable (Int) -> ByteLevelDecodingPiece)? {
        base.incrementalByteLevelDecoder
    }
    var bosToken: String? { base.bosToken }
    var eosToken: String? { base.eosToken }
    var unknownToken: String? { base.unknownToken }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        base.encode(text: text, addSpecialTokens: addSpecialTokens)
    }
    func encodeOrdinaryText(_ text: String) throws -> [Int] { try base.encodeOrdinaryText(text) }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        base.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }
    func convertTokenToId(_ token: String) -> Int? { base.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { base.convertIdToToken(id) }

    func applyChatTemplate(
        messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try applyChatTemplate(
            messages: messages, tools: tools, additionalContext: additionalContext, addGenerationPrompt: true)
    }

    func applyChatTemplate(
        messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?,
        addGenerationPrompt: Bool
    ) throws -> [Int] {
        guard !messages.isEmpty else { throw JangLoaderError.invalidConfig("ERNIE chat needs a message") }
        let context = additionalContext ?? [:]
        let disabled = context["enable_thinking"] as? Bool == false
        let rawEffort = context[Ernie55EffortPolicy.templateKey] as? String
        let effort = rawEffort.flatMap(Ernie55EffortPolicy.normalize)
        if !disabled, rawEffort != nil, effort == nil {
            throw JangLoaderError.invalidConfig("ERNIE reasoning_effort accepts low, medium or high")
        }
        var segments: [(text: String, marker: Bool)] = []
        func text(_ text: String) {
            guard !text.isEmpty else { return }
            if let last = segments.last, !last.marker {
                segments[segments.count - 1].text += text
            } else {
                segments.append((text, false))
            }
        }
        func marker(_ token: String) { segments.append((token, true)) }
        func role(_ name: String) { marker(Self.start); text(name) }
        func close() { marker(Self.end); text("\n") }
        func messageText(_ message: Message) -> String {
            if let body = message["content"] as? String { return body }
            return (message["content"] as? [Message])?.compactMap { $0["text"] as? String }.joined() ?? ""
        }

        marker(Self.mask)
        role("system\n")
        let leadingSystem = messages.first?["role"] as? String == "system"
        if leadingSystem {
            marker(Self.system)
            text("\n" + messageText(messages[0]) + "\n")
            marker(Self.systemEnd)
            text("\n")
        }
        marker(Self.settings)
        if !disabled { text("\nreasoning_effort=" + (effort ?? Ernie55EffortPolicy.defaultEffort)) }
        if let tools, !tools.isEmpty, context["parallel_tool_calls"] as? Bool != false {
            text("\nparallel_tool_calls=True")
        }
        text("\n")
        marker(Self.settingsEnd)
        if let tools, !tools.isEmpty {
            text("\n")
            marker(Self.toolList)
            text("\n[" + (try tools.map(Self.json).joined(separator: ", ")) + "]\n")
            marker(Self.toolListEnd)
        }
        close()

        let first = leadingSystem ? 1 : 0
        for index in first..<messages.count {
            let message = messages[index]
            switch message["role"] as? String {
            case "user", "system":
                role((message["role"] as! String) + "\n")
                text(messageText(message))
                close()
            case "assistant":
                role("assistant\n")
                marker(Self.think)
                if !disabled {
                    text(message["thoughts"] as? String ?? message["reasoning_content"] as? String ?? "")
                }
                marker(Self.thinkEnd)
                text(messageText(message))
                if let calls = message["tool_calls"] as? [Message], !calls.isEmpty {
                    marker(Self.toolCall)
                    text("\n")
                    for call in calls {
                        let function = call["function"] as? Message ?? call
                        guard let name = function["name"] as? String else {
                            throw JangLoaderError.invalidConfig("ERNIE tool history has no function name")
                        }
                        let arguments: [String: Any]
                        if let encoded = function["arguments"] as? String {
                            guard let parsed = try JSONSerialization.jsonObject(with: Data(encoded.utf8)) as? [String: Any]
                            else { throw JangLoaderError.invalidConfig("ERNIE tool arguments must be an object") }
                            arguments = parsed
                        } else {
                            arguments = function["arguments"] as? [String: Any] ?? [:]
                        }
                        text("<invoke name=\"\(name)\">")
                        for key in arguments.keys.sorted() {
                            let value = arguments[key]!
                            text("<parameter name=\"\(key)\">" + (try value as? String ?? Self.json(value)) + "</parameter>")
                        }
                        text("</invoke>\n")
                    }
                    marker(Self.toolCallEnd)
                }
                close()
            case "tool":
                if index == first || messages[index - 1]["role"] as? String != "tool" { role("tool") }
                text("\n")
                marker(Self.toolOutput)
                text(messageText(message))
                marker(Self.toolOutputEnd)
                if index + 1 == messages.count || messages[index + 1]["role"] as? String != "tool" { close() }
            default:
                throw JangLoaderError.invalidConfig("Unsupported ERNIE message role")
            }
        }
        if addGenerationPrompt {
            role("assistant\n")
            marker(Self.think)
            if disabled { marker(Self.thinkEnd) }
        } else if segments.last?.text == "\n" {
            segments.removeLast()
        }

        return try segments.flatMap { segment in
            if !segment.marker { return try base.encodeOrdinaryText(segment.text) }
            guard let id = base.convertTokenToId(segment.text) else {
                throw JangLoaderError.invalidConfig("Missing ERNIE token \(segment.text)")
            }
            return [id]
        }
    }

    private static func json(_ value: Any) throws -> String {
        let data = try JSONSerialization.data(
            withJSONObject: value, options: [.sortedKeys, .fragmentsAllowed, .withoutEscapingSlashes])
        var json = ""
        var quoted = false
        var escaped = false
        for character in String(decoding: data, as: UTF8.self) {
            json.append(character)
            if escaped {
                escaped = false
            } else if quoted && character == "\\" {
                escaped = true
            } else if character == "\"" {
                quoted.toggle()
            } else if !quoted && (character == ":" || character == ",") {
                json.append(" ")
            }
        }
        return json
    }
}
