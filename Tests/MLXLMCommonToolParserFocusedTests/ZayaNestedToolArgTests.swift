import Foundation
import MLXLMCommon
import Testing

/// ZAYA rows emit a fully *nested* XML tool body —
/// `<function>name … <parameter><name>key</name><value>val</value></parameter>` —
/// instead of the attribute form `<function=name><parameter=key>val</parameter>`.
/// The attribute-only scan found no `<parameter=` and dropped every argument
/// (live: `file_write` parsed but `path` reported missing). These pin the nested
/// extraction and guard the attribute path against regression.
@Suite("ZAYA nested XML tool-arg extraction")
struct ZayaNestedToolArgTests {

    private let fileWrite: [[String: any Sendable]] = [
        [
            "type": "function",
            "function": [
                "name": "file_write",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "path": ["type": "string"],
                        "content": ["type": "string"],
                    ] as [String: any Sendable],
                    "required": ["path", "content"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    ]

    private func zayaParser() -> XMLFunctionParser {
        XMLFunctionParser(
            startTag: "<zyphra_tool_call>",
            endTag: "</zyphra_tool_call>",
            decodesHTMLLineBreaks: true,
            unwrapJSONQuotedStringParameters: true)
    }

    /// Exact live shape (JANGTQ4): `<function>name` with no `=`, nested params.
    @Test("nested body with <function>name extracts every argument")
    func nestedFunctionOpenerExtractsArgs() throws {
        let content = """
            <zyphra_tool_call>
            <function>file_write
            <parameter>
            <name>path</name>
            <value>out.txt</value>
            </parameter>
            <parameter>
            <name>content</name>
            <value>hello</value>
            </parameter>
            </function>
            </zyphra_tool_call>
            """
        let call = try #require(zayaParser().parse(content: content, tools: fileWrite))
        #expect(call.function.name == "file_write")
        #expect(call.function.arguments["path"] == .string("out.txt"))
        #expect(call.function.arguments["content"] == .string("hello"))
    }

    /// JANGTQ_K shape: attribute `<function=name>` opener but nested params.
    @Test("attribute function opener with nested params still extracts args")
    func attributeOpenerNestedParamsExtractsArgs() throws {
        let content = """
            <zyphra_tool_call>
            <function=file_write>
            <parameter>
            <name>path</name>
            <value>note.txt</value>
            </parameter>
            <parameter>
            <name>content</name>
            <value>hi</value>
            </parameter>
            </function>
            </zyphra_tool_call>
            """
        let call = try #require(zayaParser().parse(content: content, tools: fileWrite))
        #expect(call.function.name == "file_write")
        #expect(call.function.arguments["path"] == .string("note.txt"))
        #expect(call.function.arguments["content"] == .string("hi"))
    }

    /// VERBATIM live JANGTQ4 capture: `<name>` / `<type>` / `<value>` interleaved.
    /// The `<type>string</type>` tag sits between name and value — the pairing
    /// must skip it and still bind name→value.
    @Test("nested body with interleaved <type> tag extracts every argument")
    func nestedWithTypeTagExtractsArgs() throws {
        let content = """
            <zyphra_tool_call>
            <function=file_write>
            <parameter>
            <name>content</name>
            <type>string</type>
            <value>hi</value>
            </parameter>
            <parameter>
            <name>path</name>
            <type>string</type>
            <value>note.txt</value>
            </parameter>
            </function>
            </zyphra_tool_call>
            """
        let call = try #require(zayaParser().parse(content: content, tools: fileWrite))
        #expect(call.function.name == "file_write")
        #expect(call.function.arguments["path"] == .string("note.txt"))
        #expect(call.function.arguments["content"] == .string("hi"))
    }

    /// VERBATIM live JANGTQ4 capture under `tool_choice:required`: the function
    /// name is wrapped in a *closed* `<function>name</function>` tag (not
    /// `<function=name>`), followed by nested `<name>`/`<value>` params.
    @Test("closed <function>name</function> tag with nested values extracts args")
    func closedFunctionTagNestedValuesExtractsArgs() throws {
        let content = """
            <zyphra_tool_call>
            <function>file_write</function>
            <parameter>
            <name>path</name>
            <value>note.txt</value>
            </parameter>
            <parameter>
            <name>content</name>
            <value>hi there</value>
            </parameter>
            </function>
            </zyphra_tool_call>
            """
        let call = try #require(zayaParser().parse(content: content, tools: fileWrite))
        #expect(call.function.name == "file_write")
        #expect(call.function.arguments["path"] == .string("note.txt"))
        #expect(call.function.arguments["content"] == .string("hi there"))
    }

    /// REGRESSION GUARD: a nested body that carries names/types but NO `<value>`
    /// tags (live JANGTQ4 auto-mode: the model echoes the tool-definition
    /// skeleton) must NOT fabricate arguments — it yields the schema-validation
    /// envelope, never a partial call with mispaired values.
    @Test("nested names without values does not fabricate arguments")
    func nestedNamesWithoutValuesNoFabrication() {
        let content = """
            <zyphra_tool_call>
            <function=file_write>
            <parameter>
            <name>content</name>
            <type>string</type>
            </parameter>
            <parameter>
            <name>path</name>
            <type>string</type>
            </parameter>
            </function>
            </zyphra_tool_call>
            """
        let call = zayaParser().parse(content: content, tools: fileWrite)
        // No <value> anywhere → nested gate fails → attribute scan finds no
        // `<parameter=` → schema validation envelope (not a real call).
        if let call {
            #expect(call.function.arguments["path"] == nil)
            #expect(call.function.arguments["_error"] == .string("invalid_tool_arguments"))
        }
    }

    /// REGRESSION GUARD: the classic attribute form must be unchanged.
    @Test("attribute-style body is untouched")
    func attributeStyleUnchanged() throws {
        let content =
            "<zyphra_tool_call>\n<function=line_count>\n<parameter=text>\nhi there\n</parameter>\n</function>\n</zyphra_tool_call>"
        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "line_count",
                    "parameters": [
                        "type": "object",
                        "properties": ["text": ["type": "string"]] as [String: any Sendable],
                        "required": ["text"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]
        let call = try #require(zayaParser().parse(content: content, tools: tools))
        #expect(call.function.name == "line_count")
        #expect(call.function.arguments["text"] == .string("hi there"))
    }

    /// REGRESSION GUARD: ordinary prose (no tool markers) never becomes a call.
    @Test("plain prose yields no tool call")
    func plainProseNoCall() {
        #expect(zayaParser().parse(content: "Sure, I can help with that.", tools: fileWrite) == nil)
    }
}
