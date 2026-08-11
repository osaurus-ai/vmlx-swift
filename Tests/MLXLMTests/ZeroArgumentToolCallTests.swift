import Foundation
import Testing

@testable import MLXLMCommon

/// A tool that declares no parameters is a normal thing to ship — Osaurus
/// alone exposes `list_mailboxes`, `capabilities_discover`, `todo_read` and
/// others. Every wire format can express calling one, and every parser has to
/// accept it, because the failure is invisible: the parser returns nil, the
/// envelope is discarded as non-content, and the surface sees a turn with no
/// text and no tool work. That reaches the user as "the model returned empty
/// output after tool execution" while the model in fact emitted a correct
/// call. (`RaptorToolContinuationTests` shows it end to end on real weights.)
///
/// One row per format, using that format's own spelling of "call this tool
/// with no arguments".
@Suite("Zero-argument tool calls")
struct ZeroArgumentToolCallTests {

    static let toolName = "list_mailboxes"

    /// Schema for a tool that takes nothing, in the shape parsers receive.
    static let noParameterTool: [String: any Sendable] = [
        "type": "function",
        "function": [
            "name": toolName,
            "description": "List all mailboxes in Apple Mail.",
            "parameters": [
                "type": "object",
                "properties": [String: any Sendable](),
                "required": [String](),
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]

    static let dsml = DeepseekV4Tokens.dsml

    /// `(format, wire text)` — the text a model emits to call
    /// `list_mailboxes` with no arguments, per format.
    static let samples: [(ToolCallFormat, String)] = [
        (.json, #"<tool_call>{"name": "list_mailboxes", "arguments": {}}</tool_call>"#),
        (.xmlFunction, "<tool_call><function=list_mailboxes></function></tool_call>"),
        (.glm4, "<tool_call>list_mailboxes</tool_call>"),
        (.hunyuan, "<tool_calls><tool_call>list_mailboxes<tool_sep></tool_call></tool_calls>"),
        (.minimaxM2, #"<invoke name="list_mailboxes"></invoke>"#),
        (.atem, """
            <atem:function_calls>
            <atem:invoke name="list_mailboxes">
            </atem:invoke>
            </atem:function_calls>
            """),
        (.dsml, """
            <\(dsml)tool_calls>
            <\(dsml)invoke name="list_mailboxes">
            </\(dsml)invoke>
            </\(dsml)tool_calls>
            """),
        (.zayaXml, "<zyphra_tool_call><function=list_mailboxes></function></zyphra_tool_call>"),
        (.kimiK2, "functions.list_mailboxes:0<|tool_call_argument_begin|>{}"),
        (.llama3, #"{"name": "list_mailboxes", "parameters": {}}"#),
        (.mistral, #"[TOOL_CALLS]list_mailboxes[ARGS]{}"#),
        (.lfm2, "<|tool_call_start|>[list_mailboxes()]<|tool_call_end|>"),
        (.gemma, "call:list_mailboxes{}"),
        (.gemma4, "<|tool_call>call:list_mailboxes{}<tool_call|>"),
        (.step, "<tool_call><function=list_mailboxes></function></tool_call>"),
        (.nemotron, "<tool_call><function=list_mailboxes></function></tool_call>"),
    ]

    /// A format added without a sample here would silently skip the check.
    @Test func everyFormatHasASample() {
        let covered = Set(Self.samples.map(\.0))
        let missing = ToolCallFormat.allCases.filter { !covered.contains($0) }
        #expect(missing.isEmpty, "no zero-argument sample for: \(missing.map(\.rawValue))")
    }

    @Test("every format parses a call with no arguments", arguments: samples)
    func zeroArgumentCallSurvives(_ sample: (format: ToolCallFormat, wire: String)) {
        let parser = sample.format.createParser()
        let tools = [Self.noParameterTool]

        // `parse` is the streaming path; `parseEOS` is the end-of-generation
        // path. A format only works if the call survives whichever one its
        // transport uses, so accept either.
        let single = parser.parse(content: sample.wire, tools: tools)
        let batch = parser.parseEOS(sample.wire, tools: tools)
        let resolved = single ?? batch.first

        #expect(
            resolved != nil,
            "\(sample.format.rawValue) discarded a zero-argument call — the model's tool call vanishes and the turn looks empty")
        #expect(
            resolved?.function.name == Self.toolName,
            "\(sample.format.rawValue) parsed the wrong function name: \(resolved?.function.name ?? "<nil>")")
        #expect(
            resolved?.function.arguments.isEmpty == true,
            "\(sample.format.rawValue) invented arguments for a no-parameter tool: \(resolved?.function.arguments ?? [:])")
    }

    // MARK: - The bare-name branch must not promote prose

    /// With no `<arg_key>` to bound it, a GLM-family function name is the whole
    /// envelope body, so the acceptance rule is all that separates a real call
    /// from text the model wrapped in `<tool_call>`.
    @Test(arguments: [
        "I should call list_mailboxes now",
        "list_mailboxes(folder=INBOX)",
        "",
        "   ",
        "{\"name\": \"list_mailboxes\"}",
        String(repeating: "a", count: 65),
    ])
    func glm4RejectsNonIdentifierBodies(_ body: String) {
        let parser = GLM4ToolCallParser()
        let call = parser.parse(content: "<tool_call>\(body)</tool_call>", tools: nil)
        #expect(call == nil, "GLM4 promoted non-identifier text to a tool call: \(body)")
    }

    @Test(arguments: ["list_mailboxes", "get.time", "todo-read", "a", "tool_9"])
    func glm4AcceptsIdentifierBodies(_ name: String) {
        let parser = GLM4ToolCallParser()
        let call = parser.parse(content: "<tool_call>\(name)</tool_call>", tools: nil)
        #expect(call?.function.name == name)
        #expect(call?.function.arguments.isEmpty == true)
    }

    /// Calls that DO carry arguments must be untouched by the bare-name rule.
    @Test func glm4ArgumentCallsUnchanged() {
        let parser = GLM4ToolCallParser()
        let call = parser.parse(
            content:
                "<tool_call>list_messages<arg_key>mailbox_path</arg_key><arg_value>you@x.com/INBOX</arg_value></tool_call>",
            tools: nil)
        #expect(call?.function.name == "list_messages")
        #expect(call?.function.arguments["mailbox_path"] == .string("you@x.com/INBOX"))
    }
}
