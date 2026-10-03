// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Testing

@testable import MLXLMCommon

@Suite("XML function string framing")
struct XMLFunctionParserFramingFocusedTests {
    private var stringTools: [[String: any Sendable]] {
        tools(properties: ["content": ["type": "string"]])
    }

    private func tools(properties: [String: any Sendable]) -> [[String: any Sendable]] {
        [["type": "function", "function": [
            "name": "file_write",
            "parameters": ["type": "object", "properties": properties] as [String: any Sendable],
        ] as [String: any Sendable]]]
    }

    private func wire(_ value: String, zaya: Bool = false) -> String {
        let body = "<function=file_write><parameter=content>\(value)</parameter></function>"
        return zaya ? "<zyphra_tool_call>\(body)</zyphra_tool_call>" : "<tool_call>\(body)</tool_call>"
    }

    private func expectContent(_ call: ToolCall, _ expected: String, label: String = "") throws {
        #expect(call.function.name == "file_write")
        let argument = try #require(call.function.arguments["content"])
        guard case .string(let actual) = argument else {
            Issue.record("Expected string content: \(label)")
            return
        }
        #expect(Array(actual.utf8) == Array(expected.utf8), Comment(rawValue: label))
    }

    // Explicit wire/payload pairs: extra edge LFs are file content, not
    // transport framing. No expected value is computed by trimming the wire.
    private var framingCases: [(label: String, value: String, expected: String)] {
        [
            ("inline", "body", "body"),
            ("literal code", "\n" + #"print("\\n")"# + "\n", #"print("\\n")"#),
            ("Windows path", "\n" + #"C:\new\test"# + "\n", #"C:\new\test"#),
            ("quoted literal", "\n" + #""red\ngreen""# + "\n", #""red\ngreen""#),
            ("one framing LF", "\nbody\n", "body"),
            ("blank at each edge", "\n\n## v1.1\n- bug fixes\n\n", "\n## v1.1\n- bug fixes\n"),
            ("two blanks at each edge", "\n\n\nline\n\n\n", "\n\nline\n\n"),
            ("leading blank", "\n\nbody\n", "\nbody"),
            ("trailing blank", "\nbody\n\n", "body\n"),
            ("spaces and tabs", "\n \tbody\t \n", " \tbody\t "),
            ("indented trailing blank", "\n  indented\n\n", "  indented\n"),
            ("empty", "", ""),
            ("one LF only", "\n", ""),
            ("two LFs only", "\n\n", ""),
            ("blank only", "\n\n\n", "\n"),
            ("Unicode", "\n\n# Résumé — naïve café\n\n", "\n# Résumé — naïve café\n"),
            ("CR is payload", "\rbody\r", "\rbody\r"),
            // Removing a trailing ASCII LF must not also remove its CR.
            ("CRLF scalar boundary", "\r\nbody\r\n", "\r\nbody\r"),
            ("framed CRLF payload", "\n\r\nbody\r\n\n", "\r\nbody\r\n"),
            ("Unicode separators", "\u{0085}body\u{2028}", "\u{0085}body\u{2028}"),
        ]
    }

    @Test("native XML factory removes one ASCII LF and preserves payload bytes")
    func nativeAttributeStringFraming() throws {
        let parser = ToolCallFormat.xmlFunction.createParser()
        for fixture in framingCases {
            let call = try #require(parser.parse(content: wire(fixture.value), tools: stringTools))
            try expectContent(call, fixture.expected, label: fixture.label)
        }
    }

    @Test("schema-unknown attribute strings use the native framing contract")
    func nativeUnknownSchemaFraming() throws {
        let schemas: [[[String: any Sendable]]?] = [
            nil, [], tools(properties: [:]), tools(properties: ["content": ["description": "Text"]]),
        ]
        for schema in schemas {
            let call = try #require(ToolCallFormat.xmlFunction.createParser().parse(
                content: wire("\n\nbody\n\n"), tools: schema))
            try expectContent(call, "\nbody\n")
        }
    }

    @Test("existing declared string aliases use the same framing contract")
    func nativeStringAliases() throws {
        for type in ["string", "STRING", "str", "text", "varchar", "char", "enum"] {
            let call = try #require(ToolCallFormat.xmlFunction.createParser().parse(
                content: wire("\n\nbody\n\n"), tools: tools(properties: ["content": ["type": type]])))
            try expectContent(call, "\nbody\n", label: type)
        }
    }

    @Test("direct XML constructors retain legacy framing")
    func directConstructorsRemainLegacy() throws {
        let parser = XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
        for schema in [stringTools, []] {
            let call = try #require(parser.parse(content: wire("\n\nbody\n\n"), tools: schema))
            try expectContent(call, "body")
        }
        let blank = try #require(parser.parse(content: wire("\n\n\n"), tools: stringTools))
        try expectContent(blank, "")
    }

    @Test("MiMo literal strings take priority over the optional framing flag")
    func literalStringPolicyWins() throws {
        let parser = XMLFunctionParser(
            startTag: "<tool_call>", endTag: "</tool_call>",
            preservesLiteralStringValues: true, trimsSingleLFStringFraming: true)
        let value = "\n\n" + #"literal\ntext"# + "\r\n"
        let call = try #require(parser.parse(content: wire(value), tools: stringTools))
        try expectContent(call, value)
    }

    @Test("typed values retain legacy framing and conversion")
    func typedValuesRemainUnchanged() throws {
        let schema = tools(properties: [
            "count": ["type": "integer"], "ratio": ["type": "number"],
            "enabled": ["type": "boolean"], "data": ["type": "object"],
            "tags": [
                "type": "array", "items": ["type": "string"] as [String: any Sendable],
            ] as [String: any Sendable],
            "content": ["type": "custom"],
        ])
        let content = "<tool_call><function=file_write>"
            + "<parameter=count>\n\n25\n\n</parameter>"
            + "<parameter=ratio>\n\n2.5\n\n</parameter>"
            + "<parameter=enabled>\n\ntrue\n\n</parameter>"
            + "<parameter=data>\n\n{\"text\":\"line\\n\"}\n\n</parameter>"
            + "<parameter=tags>\n\n[\"a\",\"b\"]\n\n</parameter>"
            + "<parameter=content>\n\nbody\n\n</parameter>"
            + "</function></tool_call>"
        for format in [ToolCallFormat.xmlFunction, .mimo] {
            let call = try #require(format.createParser().parse(content: content, tools: schema))
            #expect(call.function.arguments["count"] == .int(25))
            #expect(call.function.arguments["ratio"] == .double(2.5))
            #expect(call.function.arguments["enabled"] == .bool(true))
            #expect(call.function.arguments["data"] == .object(["text": .string("line\n")]))
            #expect(call.function.arguments["tags"] == .array([.string("a"), .string("b")]))
            try expectContent(call, "body")
        }
    }

    @Test("MiMo, ZAYA and delegated XML families keep their existing bytes")
    func otherFamiliesRemainUnchanged() throws {
        for value in ["\n", "\r\n", "env: prod\n", "\n\n# Résumé\n\n", #"literal\ntext"#] {
            let call = try #require(ToolCallFormat.mimo.createParser().parse(
                content: wire(value), tools: stringTools))
            try expectContent(call, value, label: "MiMo")
        }
        for format in [ToolCallFormat.step, .nemotron] {
            let call = try #require(format.createParser().parse(
                content: wire("\n\nbody\n\n"), tools: stringTools))
            try expectContent(call, "body", label: format.rawValue)
        }
        for format in [ToolCallFormat.zayaXml, .gemma4] {
            for fixture in [
                ("\n\nbody\n\n", "body"),
                ("\n\"red\\ngreen\\nblue\\n\"\n", "red\ngreen\nblue"),
                ("\nred<br>green<BR />blue\n", "red\ngreen\nblue"),
            ] {
                let call = try #require(format.createParser().parse(
                    content: wire(fixture.0, zaya: true), tools: stringTools))
                try expectContent(call, fixture.1, label: format.rawValue)
            }
        }
    }

    @Test("nested XML remains on its legacy value path")
    func nestedXMLRemainsUnchanged() throws {
        let content = "<tool_call><function>file_write\n<parameter><name>content</name>"
            + "<value>\n\nbody\n\n</value></parameter></function></tool_call>"
        let call = try #require(ToolCallFormat.xmlFunction.createParser().parse(
            content: content, tools: stringTools))
        try expectContent(call, "body")
    }

    @Test("JSON wire fallback retains decoded string payload without XML framing")
    func jsonFallbackRemainsUnchanged() throws {
        let content = #"<tool_call>{"name":"file_write","arguments":{"content":"\n\nbody\n\n"}}</tool_call>"#
        for format in [ToolCallFormat.xmlFunction, .mimo] {
            let call = try #require(format.createParser().parse(content: content, tools: stringTools))
            try expectContent(call, "\n\nbody\n\n")
        }
    }

    @Test("native attribute strings preserve literal code, paths and quoted text")
    func nativeStringBytes() throws {
        let cases = [
            #"literal\nnext"#,
            #"print("\\n")"#,
            #"C:\new\test"#,
            #""red\ngreen""#,
            #"quote: \" café \u0041 \b \f \/ \q"#,
            "actual\n" + #"literal\nnext"# + "\r\ttail",
            "\0\u{8}\u{C}" + #"line\nnext"#,
            "red<br>green",
        ]
        let schemas: [[[String: any Sendable]]?] = [nil, [], stringTools]
            + ["STRING", "str", "text", "varchar", "char", "enum"].map {
                tools(properties: ["content": ["type": $0]])
            }
        for value in cases {
            for schema in schemas {
                let call = try #require(ToolCallFormat.xmlFunction.createParser().parse(
                    content: wire("\n" + value + "\n"), tools: schema))
                try expectContent(call, value, label: "native literal bytes")
            }
        }
    }

    @Test("shared legacy string conversion retains its existing policy")
    func legacyStringConversionUnchanged() throws {
        let cases: [(String, String)] = [
            (#"one\ntwo\rthree\tfour"#, "one\ntwo\rthree\tfour"),
            (#"\\n"#, "\\\n"),
            ("actual\n" + #"line\nnext"#, "actual\n" + #"line\nnext"#),
            ("actual\r" + #"line\rnext"#, "actual\r" + #"line\rnext"#),
            ("actual\t" + #"line\tnext"#, "actual\t" + #"line\tnext"#),
            ("\0\u{8}\u{C}" + #"line\nnext"#, "\0\u{8}\u{C}" + #"line\nnext"#),
            (#"quote: \" café \u0041 \b \f \/ \q"#, #"quote: \" café \u0041 \b \f \/ \q"#),
        ]
        for (value, expected) in cases {
            let actual = try #require(convertParameterValue(value, paramName: "content",
                funcName: "file_write", tools: stringTools) as? String)
            #expect(Array(actual.utf8) == Array(expected.utf8))
            let unknown = try #require(convertParameterValue(value, paramName: "content",
                funcName: "file_write", tools: nil) as? String)
            #expect(Array(unknown.utf8) == Array(value.utf8))
            let literal = try #require(ToolCallFormat.mimo.createParser().parse(
                content: wire(value), tools: stringTools))
            try expectContent(literal, value, label: "MiMo mixed controls")
        }
    }

    @Test("streaming and EOS preserve the native payload at every scalar split")
    func processorChunkBoundariesPreserveBytes() throws {
        for fixture in framingCases {
            let content = wire(fixture.value)
            let scalars = content.unicodeScalars
            for split in Array(scalars.indices) + [scalars.endIndex] {
                let processor = ToolCallProcessor(format: .xmlFunction, tools: stringTools)
                var visible = processor.processChunk(String(scalars[..<split])) ?? ""
                visible += processor.processChunk(String(scalars[split...])) ?? ""
                visible += processor.processEOS() ?? ""
                #expect(visible.isEmpty, Comment(rawValue: fixture.label))
                #expect(processor.toolCalls.count == 1, Comment(rawValue: fixture.label))
                let call = try #require(processor.toolCalls.first)
                try expectContent(call, fixture.expected, label: fixture.label)
            }
        }
    }

    @Test("unknown-schema processor retains blank payload after a split parameter closer")
    func processorUnknownSchemaFraming() throws {
        let processor = ToolCallProcessor(format: .xmlFunction)
        for chunk in [
            "<tool_call><function=file_write><parameter=content>\n\n",
            "body\n\n</para", "meter></function></tool_call>",
        ] {
            _ = processor.processChunk(chunk)
        }
        _ = processor.processEOS()
        #expect(processor.toolCalls.count == 1)
        try expectContent(try #require(processor.toolCalls.first), "\nbody\n")
    }
}
