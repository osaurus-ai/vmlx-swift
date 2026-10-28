# Structured tool_calls in Chat.Message

**Landed:** 2026-04-21
**Scope:** `Libraries/MLXLMCommon/Chat.swift`
**Motivation:** Close the multi-turn-tool-call crash osaurus hit on
MiniMax, Llama 3.1/3.2, Qwen 2.5 Instruct, Mistral Large, and every
other model whose Jinja chat template reads
`message.tool_calls[i]`.

## The bug

1. osaurus's `ModelRuntime.makeMLXChatMessages`
   (`Packages/OsaurusCore/Services/ModelRuntime.swift:745-749`)
   serialized assistant tool calls into the `content` string as
   `<tool_call>{...}</tool_call>` XML because ``Chat.Message`` had no
   slot for structured tool calls.
2. ``Chat.Message`` carried only `role`, `content`, `images`,
   `videos` — no `tool_calls`, no `tool_call_id`.
3. ``DefaultMessageGenerator`` emitted only `{role, content}` into
   the Jinja-renderer dict.
4. MiniMax's
   `~/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M2.7/.../chat_template.jinja`
   reads `message.tool_calls[-1].name` on assistant messages. Since
   step 3 never emitted `tool_calls`, `last_tool_call.name` stayed
   `none`.
5. On any follow-up tool-role message, the same template then raised:
   ```
   {%- if last_tool_call.name is none -%}
     {{ raise_exception("Message has tool role, but there was no previous assistant message with a tool call!") }}
   ```

Hard failure on every multi-turn tool-call conversation.

## The fix

### `Chat.Message` — new fields

```swift
public var toolCalls: [ToolCall]?   // assistant role only
public var toolCallId: String?      // tool role only
```

Both default to `nil`; existing call sites (plain chat) compile
unchanged.

### New constructors

```swift
// Assistant that issued one or more tool calls
Chat.Message.assistant(_ content: String, toolCalls: [ToolCall])

// Tool-role reply with the originating call's id
Chat.Message.tool(_ content: String, toolCallId: String? = nil)
```

The old single-argument `assistant(_:)` and `tool(_:)` forms still
work — overload resolution picks them when no tool-call payload is
supplied.

### `DefaultMessageGenerator` — dual-view emission

Every tool call entry carries BOTH views:

```swift
[
  "id": "call_0_get_weather",        // stable per message
  "type": "function",
  // Flat view — MiniMax, Llama 3.1 Groq, others.
  "name": "get_weather",
  "arguments": ["location": "NYC"],
  // Nested view — OpenAI, HuggingFace canonical.
  "function": [
    "name": "get_weather",
    "arguments": ["location": "NYC"],
  ],
]
```

**Why both:** real chat templates in the wild read tool calls
differently.

- MiniMax line 121: `message.tool_calls[-1].name` — reads `.name`
  directly off the unbinded entry. Requires flat view.
- MiniMax line 107-108: `{%- if tool_call.function %} {%- set
  tool_call = tool_call.function %}` — handles both but prefers
  nested when present.
- OpenAI / HF canonical: `tool_calls[i].function.name`,
  `tool_calls[i].function.arguments` — requires nested view.
- Llama 3.1 Groq-style tool-use variants: `tool_calls[i].name`,
  `tool_calls[i].arguments` — requires flat view.

Emitting both is the only way to render correctly across every
template without per-model forking. Redundant bytes are cheap —
per-template engine branches are not.

### Shared dict construction

`defaultMessageDict(for:)` is a top-level `public` free function so
any custom generator can wrap it and then mutate model-specific fields
on top, without having to reimplement the tool-call emission logic.

## Usage — osaurus side

Replace the current `makeMLXChatMessages` XML-stuffing path with:

```swift
// Was: content = "<tool_call>{...}</tool_call>"
// Now:
let toolCalls: [ToolCall] = openAICall.toolCalls.map { tc in
    ToolCall(function: .init(
        name: tc.function.name,
        arguments: parseJSONArgs(tc.function.arguments)  // JSON string → [String: JSONValue]
    ))
}
let mlxMessage = Chat.Message.assistant(
    openAICall.content ?? "",
    toolCalls: toolCalls
)

// And for tool-role replies:
let mlxTool = Chat.Message.tool(
    openAITool.content,
    toolCallId: openAITool.toolCallId
)
```

## Coverage

- 10 new unit tests in
  `Tests/MLXLMTests/ChatMessageToolCallTests.swift`:
  - `Chat.Message` constructor coverage (assistant+toolCalls,
    tool+toolCallId, default-nil toolCallId)
  - Plain user message emits only role + content
  - Dual-view tool-call dict emission
  - Tool reply emits `tool_call_id`, omits when nil
  - Multiple tool calls get distinct ids
  - `DefaultMessageGenerator` passes tool_calls through
  - `NoSystemMessageGenerator` drops system but preserves tool_calls

- 129-test regression sweep across 10 suites: Chat tool-call (10),
  ReasoningParser (61 incl. Harmony + Qwen3 startInReasoning +
  edge cases A1-A8 + B1-B5), StopStringMatcher (14),
  ToolCallEdgeCases (24), CacheCoordinatorKVPolicy (10),
  SSMReDeriver (10). All green.

## Backwards compatibility

- Every existing call site compiles unchanged — the new fields
  default to `nil` and the existing factories (`.assistant(_:)`,
  `.tool(_:)` without `toolCallId:`) keep their original
  one-argument shapes.
- The Jinja-rendered dict includes `tool_calls` / `tool_call_id`
  ONLY when the message sets them, so templates that don't read
  those keys see the same dict they saw before.
- `DefaultMessageGenerator.generate(message:)` signature is
  identical — only the contents of the returned dict grew.

## Not in scope (tracked as follow-ups)

- Custom per-model generators that need to override the tool-call
  shape (e.g. a future model that wants a flat-only or nested-only
  convention) can wrap `defaultMessageDict(for:)` and post-process.
- Threading a caller-supplied `tool_call_id` through `ToolCall`
  itself (today the id is synthesized from `call_<index>_<name>`).
  When a future caller needs a specific id format — e.g., to match an
  OpenAI `id` field received over the wire — add an `id: String?`
  field to `ToolCall` and thread it through the dict.
