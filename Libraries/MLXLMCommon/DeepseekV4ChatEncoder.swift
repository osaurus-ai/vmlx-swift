// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4 chat-template encoder — Swift port of
// `jang-tools/jang_tools/dsv4_prune/encoding/encoding_dsv4.py`
// (authoritative copy also shipped inside every DSV4 bundle at
// `encoding/encoding_dsv4.py`).
//
// DSV4 does NOT ship a `chat_template` in `tokenizer_config.json` —
// callers must use this encoder to build prompts. The encoder handles:
//
//   - BOS (`<｜begin▁of▁sentence｜>`) and EOS (`<｜end▁of▁sentence｜>`)
//   - User / Assistant / System / Developer / latest_reminder roles
//   - Two thinking modes: `chat` (empty closed `</think>` tail, model
//     generates direct answer) vs `thinking` (open `<think>` tail,
//     model generates reasoning then answer)
//   - Three reasoning-effort levels: nil / "high" / "max"
//   - `drop_earlier_reasoning`: multi-turn strips prior turns' think
//     blocks (unless any message still carries `tools`)
//   - Tool call encoding in DSML format (curly-quote U+FF5C markers)
//   - Tool result merging (`<tool_result>…</tool_result>`) into user
//     messages via `content_blocks`, which DSV4 has in place of the
//     standalone `tool` role
//
// Reference: `jang/research/DSV4-RUNTIME-ARCHITECTURE.md` §4
// ("Chat template + reasoning modes") and §11 (testing matrix).

import Foundation

// MARK: - Special tokens (curly-quote markers are U+FF5C, NOT ASCII `|`)

public enum DeepseekV4Tokens {
    public static let bos = "<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}>"
    public static let eos = "<\u{FF5C}end\u{2581}of\u{2581}sentence\u{FF5C}>"
    public static let user = "<\u{FF5C}User\u{FF5C}>"
    public static let assistant = "<\u{FF5C}Assistant\u{FF5C}>"
    public static let latestReminder = "<\u{FF5C}latest_reminder\u{FF5C}>"
    public static let thinkStart = "<think>"
    public static let thinkEnd = "</think>"
    /// Curly-quote prefix/suffix chunk used inside DSML blocks. Note
    /// the tag pattern is `<｜DSML｜something>` — the markers are
    /// U+FF5C (fullwidth vertical bar), not ASCII `|`.
    public static let dsml = "\u{FF5C}DSML\u{FF5C}"

    /// Task special tokens for internal classification workflows.
    /// Kept here for completeness; the primary chat path doesn't use them.
    public static let taskSPTokens: [String: String] = [
        "action": "<\u{FF5C}action\u{FF5C}>",
        "query": "<\u{FF5C}query\u{FF5C}>",
        "authority": "<\u{FF5C}authority\u{FF5C}>",
        "domain": "<\u{FF5C}domain\u{FF5C}>",
        "title": "<\u{FF5C}title\u{FF5C}>",
        "read_url": "<\u{FF5C}read_url\u{FF5C}>",
    ]
}

// MARK: - Public enums

public enum DeepseekV4ThinkingMode: String, Sendable {
    case chat
    case thinking
}

public enum DeepseekV4ReasoningEffort: String, Sendable {
    case high
    case max
}

// MARK: - Encoder

public struct DeepseekV4ChatEncoder: Sendable {

    public init() {}

    // The REASONING_EFFORT_MAX system-level preface is a verbatim port
    // of `REASONING_EFFORT_MAX` in encoding_dsv4.py — changing the
    // string would shift prompt distribution and degrade thinking-mode
    // quality.
    static let reasoningEffortMaxPreface = """
        Reasoning Effort: Absolute maximum with no shortcuts permitted.
        You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.
        Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.


        """

    // MARK: - Public API

    /// Encode a list of `Message` structs into the DSV4 prompt format.
    /// - Parameters:
    ///   - messages: conversation history, OpenAI-style roles + fields.
    ///   - thinkingMode: `.chat` or `.thinking` (DSV4 has these two).
    ///   - reasoningEffort: nil / `.high` / `.max`. Only applies in
    ///     `.thinking` mode.
    ///   - dropEarlierReasoning: strip `reasoning_content` from prior
    ///     assistant turns. Forced `false` whenever any message still
    ///     carries `tools` (mid-trajectory agent turns need full CoT).
    ///   - context: optional pre-rendered prefix (for cached contexts);
    ///     not prepended to output but used for index accounting.
    ///   - addBOS: prepend `<｜begin▁of▁sentence｜>` when no context.
    public func encode(
        messages: [Message],
        thinkingMode: DeepseekV4ThinkingMode = .thinking,
        reasoningEffort: DeepseekV4ReasoningEffort? = nil,
        dropEarlierReasoning: Bool = true,
        context: [Message] = [],
        addBOS: Bool = true,
        toolChoiceRequired: Bool = false,
        toolChoiceName: String? = nil
    ) -> String {
        // Preprocess: merge tool messages into user, sort tool_result
        // blocks by the order they were called in the prior assistant.
        let compactedMessages = toolChoiceRequired
            ? Self.compactRequiredToolChoiceHistory(messages)
            : messages
        var processedMessages = Self.mergeToolMessages(compactedMessages)
        var processedContext = Self.mergeToolMessages(context)
        let merged = Self.sortToolResultsByCallOrder(processedContext + processedMessages)
        processedMessages = Array(merged[processedContext.count...])
        processedContext = Array(merged[..<processedContext.count])

        let full = processedContext + processedMessages

        var prompt = (addBOS && context.isEmpty) ? DeepseekV4Tokens.bos : ""

        // Resolve drop_thinking: if any message still carries tools,
        // do NOT strip (mid-agent trajectory needs reasoning).
        var effectiveDrop = dropEarlierReasoning
        if full.contains(where: { !($0.tools?.isEmpty ?? true) }) {
            effectiveDrop = false
        }

        let rendered: [Message]
        let contextLen: Int
        if thinkingMode == .thinking && effectiveDrop {
            rendered = Self.dropThinkingMessages(full)
            contextLen = rendered.count - processedMessages.count
        } else {
            rendered = full
            contextLen = processedContext.count
        }

        var finalRendered = rendered
        if toolChoiceRequired,
            finalRendered.contains(where: { !($0.tools?.isEmpty ?? true) }),
            finalRendered.last?.role != .latestReminder
        {
            let requiredToolName =
                Self.normalizedToolChoiceName(toolChoiceName)
                ?? Self.singleAvailableToolName(in: finalRendered)
            let reminder = Self.requiredToolChoiceReminder(toolName: requiredToolName)
            if let tailIndex = finalRendered.lastIndex(where: {
                $0.role == .user || $0.role == .developer
            }), tailIndex >= contextLen {
                let insertionIndex = finalRendered[tailIndex].task == nil
                    ? finalRendered.index(after: tailIndex)
                    : tailIndex
                finalRendered.insert(reminder, at: insertionIndex)
            } else {
                finalRendered.append(reminder)
            }
        }

        for i in 0..<(finalRendered.count - contextLen) {
            prompt += renderMessage(
                at: i + contextLen,
                in: finalRendered,
                thinkingMode: thinkingMode,
                dropThinking: effectiveDrop,
                reasoningEffort: reasoningEffort,
                toolChoiceRequired: toolChoiceRequired,
                toolChoiceName: toolChoiceName
            )
        }

        return prompt
    }

    // MARK: - Message rendering

    func renderMessage(
        at index: Int,
        in messages: [Message],
        thinkingMode: DeepseekV4ThinkingMode,
        dropThinking: Bool,
        reasoningEffort: DeepseekV4ReasoningEffort?,
        toolChoiceRequired: Bool,
        toolChoiceName: String?
    ) -> String {
        let msg = messages[index]
        let lastUserIdx = Self.findLastUserIndex(messages)
        var out = ""

        // Reasoning-effort preface only at index 0 in thinking mode.
        if index == 0 && thinkingMode == .thinking && reasoningEffort == .max {
            out += Self.reasoningEffortMaxPreface
        }

        switch msg.role {
        case .system:
            out += msg.content ?? ""
            if let tools = msg.tools, !tools.isEmpty {
                out += "\n\n" + Self.renderTools(
                    tools,
                    toolChoiceRequired: toolChoiceRequired,
                    toolChoiceName: toolChoiceName
                )
            }
            if let rf = msg.responseFormat {
                out += "\n\n" + Self.renderResponseFormat(rf)
            }
            if index + 1 < messages.count {
                out += "\n"
            }

        case .developer:
            var s = DeepseekV4Tokens.user
            s += msg.content ?? ""
            if let tools = msg.tools, !tools.isEmpty {
                s += "\n\n" + Self.renderTools(
                    tools,
                    toolChoiceRequired: toolChoiceRequired,
                    toolChoiceName: toolChoiceName
                )
            }
            if let rf = msg.responseFormat {
                s += "\n\n" + Self.renderResponseFormat(rf)
            }
            out += s

        case .user:
            out += DeepseekV4Tokens.user
            if let blocks = msg.contentBlocks, !blocks.isEmpty {
                var parts: [String] = []
                for b in blocks {
                    switch b {
                    case .text(let s): parts.append(s)
                    case .toolResult(_, let content):
                        parts.append("<tool_result>\(content)</tool_result>")
                    }
                }
                out += parts.joined(separator: "\n\n")
            } else {
                out += msg.content ?? ""
            }

        case .latestReminder:
            out += DeepseekV4Tokens.latestReminder + (msg.content ?? "")

        case .tool:
            // DSV4 merges tool messages into user; caller should
            // preprocess. We never reach here after mergeToolMessages.
            break

        case .assistant:
            var thinkingPart = ""
            var toolCallContent = ""

            // Render tool calls (if any) in DSML format.
            if let tcs = msg.toolCalls, !tcs.isEmpty {
                let blocks = tcs.map { tc in
                    Self.renderToolCallInvoke(name: tc.name, arguments: tc.arguments)
                }
                let joined = blocks.joined(separator: "\n")
                toolCallContent =
                    "\n\n<\(DeepseekV4Tokens.dsml)tool_calls>\n\(joined)\n</\(DeepseekV4Tokens.dsml)tool_calls>"
            }

            let content = msg.content ?? ""
            let reasoning = msg.reasoningContent ?? ""

            // A previous user message with a `task` means this is a
            // task-output assistant turn — no thinking block even in
            // thinking mode.
            let prevHasTask = index - 1 >= 0 && messages[index - 1].task != nil

            if thinkingMode == .thinking && !prevHasTask {
                if !dropThinking || index > lastUserIdx {
                    thinkingPart = reasoning + DeepseekV4Tokens.thinkEnd
                }
            }

            var piece = thinkingPart + content + toolCallContent
            if !msg.woEOS {
                piece += DeepseekV4Tokens.eos
            }
            out += piece
        }

        // Append transition tokens (Assistant marker + opening/closing
        // `<think>`/`</think>`) when the current message is the tail
        // of a user/developer turn OR there's a following assistant.
        let nextRole: MessageRole? =
            (index + 1 < messages.count) ? messages[index + 1].role : nil
        if nextRole != nil && nextRole != .assistant {
            return out
        }

        if let task = msg.task {
            let taskSP = DeepseekV4Tokens.taskSPTokens[task] ?? ""
            if task != "action" {
                out += taskSP
            } else {
                out += DeepseekV4Tokens.assistant
                out += thinkingMode == .thinking
                    ? DeepseekV4Tokens.thinkStart
                    : DeepseekV4Tokens.thinkEnd
                out += taskSP
            }
        } else if msg.role == .user || msg.role == .developer || msg.role == .latestReminder {
            out += DeepseekV4Tokens.assistant
            // Decision tree for the final tail tag:
            //   dropThinking=false + thinking → `<think>` (open)
            //   dropThinking=true  + thinking + index>=lastUser → `<think>` (open)
            //   otherwise → `</think>` (closed, chat mode)
            if !dropThinking && thinkingMode == .thinking {
                out += DeepseekV4Tokens.thinkStart
            } else if dropThinking && thinkingMode == .thinking && index >= lastUserIdx {
                out += DeepseekV4Tokens.thinkStart
            } else {
                out += DeepseekV4Tokens.thinkEnd
            }
        }

        return out
    }

    // MARK: - Tools + schema rendering

    static let toolsTemplate = """
        ## Tools

        You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<\(DeepseekV4Tokens.dsml)tool_calls>" block like the following:

        <\(DeepseekV4Tokens.dsml)tool_calls>
        <\(DeepseekV4Tokens.dsml)invoke name="$TOOL_NAME">
        <\(DeepseekV4Tokens.dsml)parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</\(DeepseekV4Tokens.dsml)parameter>
        ...
        </\(DeepseekV4Tokens.dsml)invoke>
        <\(DeepseekV4Tokens.dsml)invoke name="$TOOL_NAME2">
        ...
        </\(DeepseekV4Tokens.dsml)invoke>
        </\(DeepseekV4Tokens.dsml)tool_calls>

        For tools with no parameters, emit an empty invoke with no parameter lines:

        <\(DeepseekV4Tokens.dsml)tool_calls>
        <\(DeepseekV4Tokens.dsml)invoke name="$TOOL_NAME_WITHOUT_PARAMETERS">
        </\(DeepseekV4Tokens.dsml)invoke>
        </\(DeepseekV4Tokens.dsml)tool_calls>

        Do not emit JSON objects for tool calls; tool calls must use DSML invoke blocks.

        String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

        If thinking_mode is enabled (triggered by \(DeepseekV4Tokens.thinkStart)), you MUST output your complete reasoning inside \(DeepseekV4Tokens.thinkStart)...\(DeepseekV4Tokens.thinkEnd) BEFORE any tool calls or final response.

        Otherwise, output directly after \(DeepseekV4Tokens.thinkEnd) with tool calls or final response.

        ### Available Tool Schemas

        %@

        You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
        """

    static func renderTools(
        _ tools: [[String: any Sendable]],
        toolChoiceRequired: Bool = false,
        toolChoiceName: String? = nil
    ) -> String {
        let schemas = tools.map { Self.functionSpec(from: $0) }
        let schemaJson = schemas.map { $0.jsonSerialized() }
        let schemasBlock = schemaJson.joined(separator: "\n")
        var rendered = toolsTemplate.replacingOccurrences(of: "%@", with: schemasBlock)
        if toolChoiceRequired {
            let requiredToolName =
                Self.normalizedToolChoiceName(toolChoiceName)
                ?? Self.singleAvailableToolName(in: tools)
            let namedInstruction = requiredToolName.map {
                " Use the `\($0)` function."
            } ?? ""
            rendered +=
                "\n\nThe current assistant response MUST be a tool call. "
                + "Start with a \"<\(DeepseekV4Tokens.dsml)tool_calls>\" block "
                + "and do not answer in prose before the tool result."
                + namedInstruction
        }
        return rendered
    }

    static func requiredToolChoiceReminder(toolName: String? = nil) -> Message {
        let namedInstruction = toolName.map {
            " Use the `\($0)` function."
        } ?? ""
        return Message(
            role: .latestReminder,
            content:
                "The active API tool_choice is required for this assistant turn. "
                + "Call exactly one available tool now using a <\(DeepseekV4Tokens.dsml)tool_calls> block. "
                + "Do not answer in prose before the tool result."
                + namedInstruction
        )
    }

    static func normalizedToolChoiceName(_ name: String?) -> String? {
        guard let name else { return nil }
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    static func singleAvailableToolName(in messages: [Message]) -> String? {
        var names: Set<String> = []
        for message in messages {
            guard let tools = message.tools else { continue }
            for tool in tools {
                if let name = Self.functionSpec(from: tool)["name"] as? String,
                    let normalized = Self.normalizedToolChoiceName(name)
                {
                    names.insert(normalized)
                }
            }
        }
        return names.count == 1 ? names.first : nil
    }

    static func singleAvailableToolName(in tools: [[String: any Sendable]]) -> String? {
        var names: Set<String> = []
        for tool in tools {
            if let name = Self.functionSpec(from: tool)["name"] as? String,
                let normalized = Self.normalizedToolChoiceName(name)
            {
                names.insert(normalized)
            }
        }
        return names.count == 1 ? names.first : nil
    }

    static func renderResponseFormat(_ rf: [String: any Sendable]) -> String {
        let json = rf.jsonSerialized()
        return
            "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n\(json)"
    }

    /// Extract the `function` sub-dict from OpenAI-format tool specs.
    /// Matches `tools_from_openai_format` in encoding_dsv4.py — when
    /// a tool is wrapped in `{"type":"function","function":{…}}` we
    /// peel the wrapper; otherwise pass through.
    static func functionSpec(from tool: [String: any Sendable]) -> [String: any Sendable] {
        if let fn = tool["function"] as? [String: any Sendable] {
            return fn
        }
        return tool
    }

    // MARK: - Tool-call invoke rendering (DSML encode)

    /// Render a single `<｜DSML｜invoke>` block for one tool call.
    /// `arguments` is a JSON string (OpenAI convention) or already-decoded
    /// dict. Python reference accepts either.
    public static func renderToolCallInvoke(name: String, arguments: String) -> String {
        let params: [String: Any]
        if let data = arguments.data(using: .utf8),
            let parsed = try? JSONSerialization.jsonObject(with: data, options: [])
                as? [String: Any]
        {
            params = parsed
        } else {
            // Malformed JSON — wrap raw into an "arguments" field so
            // the DSML envelope still round-trips.
            params = ["arguments": arguments]
        }
        return renderToolCallInvoke(name: name, params: params)
    }

    public static func renderToolCallInvoke(name: String, params: [String: Any]) -> String {
        var paramBlocks: [String] = []
        // Encode each param — string params carry `string="true"` and
        // a raw value; all other JSON types serialize as JSON with
        // `string="false"`.
        for (k, v) in params {
            let isString = v is String
            let value: String
            if isString, let s = v as? String {
                value = s
            } else {
                value =
                    (try? String(
                        data: JSONSerialization.data(
                            withJSONObject: v, options: [.fragmentsAllowed, .withoutEscapingSlashes]
                        ),
                        encoding: .utf8)) ?? "\(v)"
            }
            let line =
                "<\(DeepseekV4Tokens.dsml)parameter name=\"\(k)\" string=\"\(isString ? "true" : "false")\">\(value)</\(DeepseekV4Tokens.dsml)parameter>"
            paramBlocks.append(line)
        }
        let inner = paramBlocks.joined(separator: "\n")
        return
            "<\(DeepseekV4Tokens.dsml)invoke name=\"\(name)\">\n\(inner)\n</\(DeepseekV4Tokens.dsml)invoke>"
    }

    // MARK: - Multi-turn helpers (drop_thinking, tool-merge)

    /// Strip `reasoning_content` from all assistant messages BEFORE the
    /// last user message. Keep user/system/tool/latest_reminder always.
    /// Mirrors `_drop_thinking_messages` in encoding_dsv4.py.
    static func dropThinkingMessages(_ messages: [Message]) -> [Message] {
        let lastUserIdx = findLastUserIndex(messages)
        let keepRoles: Set<MessageRole> = [.user, .system, .tool, .latestReminder]
        var result: [Message] = []
        for (idx, msg) in messages.enumerated() {
            if keepRoles.contains(msg.role) || idx >= lastUserIdx {
                result.append(msg)
            } else if msg.role == .assistant {
                var copy = msg
                copy.reasoningContent = nil
                result.append(copy)
            }
            // developer before last user: drop entirely.
        }
        return result
    }

    /// Merge `role:"tool"` messages into the preceding user message via
    /// `content_blocks`. DSV4 does NOT carry a standalone tool role.
    /// Mirrors `merge_tool_messages` in encoding_dsv4.py.
    static func mergeToolMessages(_ messages: [Message]) -> [Message] {
        var merged: [Message] = []
        for raw in messages {
            var m = raw
            switch m.role {
            case .tool:
                let block = MessageContentBlock.toolResult(
                    toolUseId: m.toolCallId ?? "", content: m.content ?? "")
                if var last = merged.last, last.role == .user,
                    var blocks = last.contentBlocks, !blocks.isEmpty
                {
                    blocks.append(block)
                    last.contentBlocks = blocks
                    merged[merged.count - 1] = last
                } else {
                    merged.append(
                        Message(role: .user, contentBlocks: [block]))
                }
            case .user:
                let textBlock = MessageContentBlock.text(m.content ?? "")
                if var last = merged.last,
                    last.role == .user,
                    var blocks = last.contentBlocks,
                    !blocks.isEmpty,
                    last.task == nil
                {
                    blocks.append(textBlock)
                    last.contentBlocks = blocks
                    last.task = m.task
                    last.tools = m.tools ?? last.tools
                    last.responseFormat = m.responseFormat ?? last.responseFormat
                    last.woEOS = m.woEOS
                    merged[merged.count - 1] = last
                } else {
                    m.contentBlocks = [textBlock]
                    merged.append(m)
                }
            default:
                merged.append(m)
            }
        }
        return merged
    }

    /// Sort `tool_result` blocks inside user messages by the order the
    /// calls appear in the preceding assistant turn. Python has the
    /// same behavior — needed because OpenAI-style tool responses can
    /// arrive out of order.
    static func sortToolResultsByCallOrder(_ messages: [Message]) -> [Message] {
        var lastCallOrder: [String: Int] = [:]
        var out: [Message] = []
        for raw in messages {
            var msg = raw
            if msg.role == .assistant, let tcs = msg.toolCalls, !tcs.isEmpty {
                lastCallOrder.removeAll()
                for (idx, tc) in tcs.enumerated() {
                    if let id = tc.id {
                        lastCallOrder[id] = idx
                    }
                }
            } else if msg.role == .user, var blocks = msg.contentBlocks, !blocks.isEmpty {
                let toolBlocks = blocks.compactMap { block -> (Int, MessageContentBlock)? in
                    if case .toolResult(let id, _) = block {
                        return (lastCallOrder[id] ?? Int.max, block)
                    }
                    return nil
                }
                if toolBlocks.count > 1 && !lastCallOrder.isEmpty {
                    let sorted = toolBlocks.sorted { $0.0 < $1.0 }.map { $0.1 }
                    var sortedIdx = 0
                    var newBlocks: [MessageContentBlock] = []
                    for block in blocks {
                        if case .toolResult = block {
                            newBlocks.append(sorted[sortedIdx])
                            sortedIdx += 1
                        } else {
                            newBlocks.append(block)
                        }
                    }
                    blocks = newBlocks
                    msg.contentBlocks = blocks
                }
            }
            out.append(msg)
        }
        return out
    }

    /// DSV4's required-tool template is sensitive to an ordinary prose
    /// assistant answer sitting between a previous tool result and the next
    /// required tool turn: live rows showed it can terminate with a blank
    /// assistant response instead of entering DSML. For required tool-choice
    /// turns, collapse that completed prose exchange so the prior tool result
    /// merges directly with the latest user request. This preserves the
    /// actual tool result and latest user instruction; it does not synthesize
    /// a tool call or alter sampler behavior.
    static func compactRequiredToolChoiceHistory(_ messages: [Message]) -> [Message] {
        guard
            let finalUserIndex = messages.indices.last(where: {
                messages[$0].role == .user || messages[$0].role == .developer
            }),
            finalUserIndex == messages.indices.last,
            let lastToolIndex = messages[..<finalUserIndex].indices.last(where: {
                messages[$0].role == .tool
            }),
            finalUserIndex > lastToolIndex + 1,
            messages[(lastToolIndex + 1)..<finalUserIndex].contains(where: {
                $0.role == .assistant && ($0.toolCalls?.isEmpty ?? true)
            })
        else {
            return messages
        }

        var compacted = Array(messages[...lastToolIndex])
        compacted.append(messages[finalUserIndex])
        return compacted
    }

    static func findLastUserIndex(_ messages: [Message]) -> Int {
        for idx in (0..<messages.count).reversed() {
            let r = messages[idx].role
            if r == .user || r == .developer { return idx }
        }
        return -1
    }
}

// MARK: - DSV4-scoped message types
//
// Namespaced inside `DeepseekV4ChatEncoder` to avoid clashing with the
// existing top-level `Chat.Message` / `ToolCall` types in MLXLMCommon.
// The DSV4 encoder has extra fields (reasoningContent, contentBlocks,
// task, woEOS) that Chat.Message doesn't carry, so keeping them
// separate is cleaner than inflating the general Chat type.

extension DeepseekV4ChatEncoder {

    public enum MessageRole: String, Sendable, Codable, Hashable {
        case system
        case developer
        case user
        case assistant
        case tool
        case latestReminder = "latest_reminder"
    }

    public enum MessageContentBlock: Sendable, Hashable {
        case text(String)
        case toolResult(toolUseId: String, content: String)
    }

    public struct ToolCall: Sendable, Hashable {
        public let id: String?
        public let name: String
        public let arguments: String  // JSON string per OpenAI convention

        public init(id: String? = nil, name: String, arguments: String) {
            self.id = id
            self.name = name
            self.arguments = arguments
        }
    }

    public struct Message: Sendable {
        public var role: MessageRole
        public var content: String?
        public var contentBlocks: [MessageContentBlock]?
        public var reasoningContent: String?
        public var toolCalls: [ToolCall]?
        public var toolCallId: String?
        public var tools: [[String: any Sendable]]?
        public var responseFormat: [String: any Sendable]?
        public var task: String?
        public var woEOS: Bool

        public init(
            role: MessageRole,
            content: String? = nil,
            contentBlocks: [MessageContentBlock]? = nil,
            reasoningContent: String? = nil,
            toolCalls: [ToolCall]? = nil,
            toolCallId: String? = nil,
            tools: [[String: any Sendable]]? = nil,
            responseFormat: [String: any Sendable]? = nil,
            task: String? = nil,
            woEOS: Bool = false
        ) {
            self.role = role
            self.content = content
            self.contentBlocks = contentBlocks
            self.reasoningContent = reasoningContent
            self.toolCalls = toolCalls
            self.toolCallId = toolCallId
            self.tools = tools
            self.responseFormat = responseFormat
            self.task = task
            self.woEOS = woEOS
        }
    }
}

/// Short type aliases kept top-level so call sites inside this file
/// don't have to spell out the DSV4 namespace on every reference.
/// External callers always go through `DeepseekV4ChatEncoder.Message`
/// etc. — these aliases are private to the file.
typealias DSV4Message = DeepseekV4ChatEncoder.Message
typealias DSV4MessageRole = DeepseekV4ChatEncoder.MessageRole
typealias DSV4ContentBlock = DeepseekV4ChatEncoder.MessageContentBlock
typealias DSV4ToolCall = DeepseekV4ChatEncoder.ToolCall

// MARK: - JSON serialization helper

private extension Dictionary where Key == String, Value == any Sendable {
    func jsonSerialized() -> String {
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: self, options: [.withoutEscapingSlashes])
        else { return "{}" }
        return String(data: data, encoding: .utf8) ?? "{}"
    }
}
