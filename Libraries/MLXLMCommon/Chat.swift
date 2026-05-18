// Copyright © 2025 Apple Inc.

public enum Chat {
    public struct Message {
        /// The role of the message sender.
        public var role: Role

        /// The content of the message.
        public var content: String

        /// Optional assistant reasoning text carried on OpenAI-compatible
        /// multi-turn messages as `reasoning_content`. Several reasoning
        /// chat templates (ZAYA, Nemotron-H/Omni, MiniMax) read this field
        /// directly and reconstruct the prior `<think>...</think>` envelope
        /// before rendering the visible assistant answer.
        public var reasoningContent: String?

        /// Array of image data associated with the message.
        public var images: [UserInput.Image]

        /// Array of video data associated with the message.
        public var videos: [UserInput.Video]

        /// Array of audio data associated with the message. Used by
        /// speech-aware models (Nemotron-3-Nano-Omni Parakeet path).
        /// Stays empty for text/image-only models. The osaurus
        /// `MessageContentPart.audioInput` → `mapOpenAIChatToMLX`
        /// path populates this field; downstream
        /// ``UserInput.init(chat:)`` copies it into ``UserInput/audios``.
        public var audios: [UserInput.Audio]

        /// Structured tool calls issued by this message (assistant role
        /// only). `nil` on messages that did not issue tool calls.
        ///
        /// ## Why this field exists
        ///
        /// Chat templates for tool-using models (MiniMax, Llama 3.x,
        /// Qwen 2.5 Instruct, Mistral Large, every OpenAI-compatible
        /// tool-using schema) read `message.tool_calls` from the Jinja
        /// dict the `MessageGenerator` produces. Before this field
        /// existed, callers had no way to communicate structured tool
        /// calls to the generator — osaurus and other hosts had to
        /// inline the XML into `content` and pray the template tolerated
        /// it. Templates that read `message.tool_calls[i].name`
        /// explicitly would raise
        /// "Message has tool role, but there was no previous assistant
        /// message with a tool call!" on any tool-role follow-up.
        ///
        /// ``DefaultMessageGenerator`` emits the contents of this field
        /// into the rendered dict under `tool_calls`, with every entry
        /// carrying BOTH a top-level `name`/`arguments` view and a
        /// nested `function.name`/`function.arguments` view — covering
        /// every convention we've seen in real templates.
        public var toolCalls: [ToolCall]?

        /// The id of the tool call this message is responding to
        /// (tool role only). OpenAI chat schema calls this
        /// `tool_call_id`; Jinja templates bind it to the originating
        /// assistant message's tool-call entry so the model knows which
        /// invocation the response belongs to.
        public var toolCallId: String?

        public init(
            role: Role, content: String, images: [UserInput.Image] = [],
            videos: [UserInput.Video] = [],
            audios: [UserInput.Audio] = [],
            reasoningContent: String? = nil,
            toolCalls: [ToolCall]? = nil,
            toolCallId: String? = nil
        ) {
            self.role = role
            self.content = content
            self.reasoningContent = reasoningContent
            self.images = images
            self.videos = videos
            self.audios = audios
            self.toolCalls = toolCalls
            self.toolCallId = toolCallId
        }

        public static func system(
            _ content: String, images: [UserInput.Image] = [],
            videos: [UserInput.Video] = [], audios: [UserInput.Audio] = []
        ) -> Self {
            Self(role: .system, content: content,
                 images: images, videos: videos, audios: audios)
        }

        /// Build an assistant message with plain text content.
        public static func assistant(
            _ content: String, images: [UserInput.Image] = [],
            videos: [UserInput.Video] = [], audios: [UserInput.Audio] = []
        ) -> Self {
            Self(role: .assistant, content: content,
                 images: images, videos: videos, audios: audios)
        }

        /// Build an assistant message that issued one or more tool
        /// calls. `content` is typically empty for a pure tool-call
        /// turn but may carry a textual explanation the model emitted
        /// alongside the call.
        public static func assistant(
            _ content: String,
            toolCalls: [ToolCall],
            images: [UserInput.Image] = [],
            videos: [UserInput.Video] = [],
            audios: [UserInput.Audio] = []
        ) -> Self {
            Self(
                role: .assistant, content: content,
                images: images, videos: videos, audios: audios,
                toolCalls: toolCalls
            )
        }

        public static func user(
            _ content: String, images: [UserInput.Image] = [],
            videos: [UserInput.Video] = [], audios: [UserInput.Audio] = []
        ) -> Self {
            Self(role: .user, content: content,
                 images: images, videos: videos, audios: audios)
        }

        /// Build a tool-role message carrying the result of a tool call.
        /// When `toolCallId` is non-nil, the generator emits
        /// `tool_call_id` into the rendered dict so the template can
        /// tie the result back to the originating assistant call.
        public static func tool(_ content: String, toolCallId: String? = nil) -> Self {
            Self(role: .tool, content: content, toolCallId: toolCallId)
        }

        public enum Role: String, Sendable {
            case user
            case assistant
            case system
            case tool
        }
    }
}

/// Protocol for something that can convert structured
/// ``Chat.Message`` into model specific ``Message``
/// (raw dictionary) format.
///
/// Typically this is owned and used by a ``UserInputProcessor``:
///
/// ```swift
/// public func prepare(input: UserInput) async throws -> LMInput {
///     let messages = Qwen2VLMessageGenerator().generate(from: input)
///     ...
/// ```
public protocol MessageGenerator: Sendable {

    /// Generates messages from the input.
    func generate(from input: UserInput) -> [Message]

    /// Returns array of `[String: any Sendable]` aka ``Message``
    func generate(messages: [Chat.Message]) -> [Message]

    /// Returns `[String: any Sendable]`, aka ``Message``.
    func generate(message: Chat.Message) -> Message
}

extension MessageGenerator {

    public func generate(message: Chat.Message) -> Message {
        defaultMessageDict(for: message)
    }

    public func generate(messages: [Chat.Message]) -> [Message] {
        var rawMessages: [Message] = []

        for message in messages {
            let raw = generate(message: message)
            rawMessages.append(raw)
        }

        return rawMessages
    }

    public func generate(from input: UserInput) -> [Message] {
        switch input.prompt {
        case .text(let text):
            generate(messages: [.user(text)])
        case .messages(let messages):
            messages
        case .chat(let messages):
            generate(messages: messages)
        }
    }
}

/// Default implementation of ``MessageGenerator`` that produces a
/// `role` + `content` dict, augmented with `tool_calls` (on assistant
/// messages that set ``Chat.Message/toolCalls``) and `tool_call_id`
/// (on tool-role messages that set ``Chat.Message/toolCallId``).
///
/// ## Dict shape
///
/// Plain text message:
/// ```swift
/// ["role": "user", "content": "hi"]
/// ```
///
/// Assistant message with a tool call:
/// ```swift
/// [
///   "role": "assistant",
///   "content": "",
///   "tool_calls": [
///     [
///       "id": "call_abc",
///       "type": "function",
///       // flat-access form for templates that read tool_calls[i].name
///       // (MiniMax chat_template.jinja, Llama 3.1 Groq format):
///       "name": "get_weather",
///       "arguments": ["location": "NYC"],
///       // nested form for OpenAI / HuggingFace-canonical templates
///       // that read tool_calls[i].function.name:
///       "function": [
///         "name": "get_weather",
///         "arguments": ["location": "NYC"],
///       ],
///     ]
///   ]
/// ]
/// ```
///
/// Tool-role reply:
/// ```swift
/// [
///   "role": "tool",
///   "content": "{\"temp_f\": 72}",
///   "tool_call_id": "call_abc",
/// ]
/// ```
///
/// Emitting both the flat (`name`/`arguments`) and nested
/// (`function.name`/`function.arguments`) views is deliberate — it
/// matches every tool-calling chat template we've seen in production
/// without requiring per-model forking. Redundant bytes are cheap;
/// per-template engine branching is not.
public struct DefaultMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> Message {
        defaultMessageDict(for: message)
    }
}

/// Implementation of ``MessageGenerator`` that produces the default
/// dict shape but omits `system` roles.
public struct NoSystemMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(messages: [Chat.Message]) -> [Message] {
        messages
            .filter { $0.role != .system }
            .map { generate(message: $0) }
    }
}

// MARK: - Default dict construction

/// Produce the canonical Jinja-renderer dict for a ``Chat.Message``.
///
/// Shared by every generator that doesn't override the individual-
/// message step — `DefaultMessageGenerator`, `NoSystemMessageGenerator`,
/// and any caller-defined generator that wants the stock shape for
/// tool-using messages without reimplementing the mapping.
///
/// Kept as a free function (not a protocol requirement) so callers can
/// compose it from custom generators: wrap the default output and then
/// mutate or add model-specific fields without having to reconstruct
/// the tool-call emission logic.
public func defaultMessageDict(for message: Chat.Message) -> Message {
    var dict: Message = [
        "role": message.role.rawValue,
        "content": message.content,
    ]

    if let reasoningContent = message.reasoningContent {
        dict["reasoning_content"] = reasoningContent
    }

    if let toolCalls = message.toolCalls, !toolCalls.isEmpty {
        dict["tool_calls"] = toolCalls.enumerated().map { (idx, call) in
            toolCallDict(call: call, index: idx)
        }
    }

    if let toolCallId = message.toolCallId {
        dict["tool_call_id"] = toolCallId
    }

    return dict
}

/// Expand a ``ToolCall`` into the OpenAI-compatible dict shape with
/// both flat and nested name/arguments views.
private func toolCallDict(call: ToolCall, index: Int) -> [String: any Sendable] {
    let name = call.function.name
    // Convert JSONValue arguments to a plain `[String: any Sendable]`
    // so the Jinja renderer sees native dicts on `.items()`.
    let args = call.function.arguments.mapValues { $0.sendableValue }

    return [
        // Synthesized unique id per message (OpenAI requires one; when
        // the caller supplied an OpenAI `tool_call.id`, preserve it so
        // later `role=tool` messages can correlate with the right call.
        "id": call.id ?? "call_\(index)_\(name)",
        "type": "function",
        // Flat view — MiniMax, Llama 3.1 Groq, others.
        "name": name,
        "arguments": args,
        // Nested view — OpenAI, HuggingFace canonical.
        "function": [
            "name": name,
            "arguments": args,
        ] as [String: any Sendable],
    ]
}
