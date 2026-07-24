import MLXLMCommon
import Testing

@Suite("LFM and DSML nested Pythonic tool arguments")
struct PythonicNestedToolArgumentTests {
    private func databaseTools() -> [ToolSpec] {
        [
            [
                "type": "function",
                "function": [
                    "name": "db_insert",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "rows": [
                                "type": "array",
                                "items": ["type": "object"] as [String: any Sendable],
                            ] as [String: any Sendable],
                            "table": ["type": "string"] as [String: any Sendable],
                            "replace": ["type": "boolean"] as [String: any Sendable],
                        ] as [String: any Sendable],
                        "required": ["rows", "table"] as [String],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]
    }

    @Test("LFM preserves nested row objects instead of splitting at inner commas")
    func lfmPreservesNestedRows() throws {
        let output =
            #"<|tool_call_start|>[db_insert(rows=[{'id': 11, 'label': 'lfm-eleven'}], table='proof_lfm')]<|tool_call_end|>"#

        let call = try #require(
            LFM2ToolCallParser().parse(content: output, tools: databaseTools()))

        #expect(call.function.name == "db_insert")
        #expect(call.function.arguments["table"] == .string("proof_lfm"))
        #expect(
            call.function.arguments["rows"]
                == .array([
                    .object([
                        "id": .int(11),
                        "label": .string("lfm-eleven"),
                    ])
                ]))
    }

    @Test("LFM preserves Unicode, booleans, nulls, numbers, and nested containers")
    func lfmPreservesTypedNestedValues() throws {
        let output =
            #"<|tool_call_start|>[db_insert(rows=[{'id': 12, 'label': 'Привет, 世界', 'active': True, 'note': None, 'metrics': {'score': 2.5}}], table='proof_unicode', replace=False)]<|tool_call_end|>"#

        let call = try #require(
            LFM2ToolCallParser().parse(content: output, tools: databaseTools()))

        #expect(call.function.arguments["replace"] == .bool(false))
        #expect(
            call.function.arguments["rows"]
                == .array([
                    .object([
                        "id": .int(12),
                        "label": .string("Привет, 世界"),
                        "active": .bool(true),
                        "note": .null,
                        "metrics": .object(["score": .double(2.5)]),
                    ])
                ]))
    }

    @Test("DSML Pythonic fallback shares the nested typed-argument contract")
    func dsmlFallbackPreservesNestedRows() throws {
        let output =
            #"db_insert(rows=[{'id': 21, 'label': 'dsml-twenty-one'}], table='proof_dsml', replace=True)"#

        let call = try #require(
            DSMLToolCallParser().parse(content: output, tools: databaseTools()))

        #expect(call.function.name == "db_insert")
        #expect(call.function.arguments["table"] == .string("proof_dsml"))
        #expect(call.function.arguments["replace"] == .bool(true))
        #expect(
            call.function.arguments["rows"]
                == .array([
                    .object([
                        "id": .int(21),
                        "label": .string("dsml-twenty-one"),
                    ])
                ]))
    }

    @Test("DSML keeps its observed JSON-label function fallback")
    func dsmlJSONLabelFallbackRemainsSupported() throws {
        let output =
            #"db_insert("rows": [{"id": 22, "label": "colon-style"}], "table": "proof_dsml_colon", "replace": true)"#

        let call = try #require(
            DSMLToolCallParser().parse(content: output, tools: databaseTools()))

        #expect(call.function.name == "db_insert")
        #expect(call.function.arguments["table"] == .string("proof_dsml_colon"))
        #expect(call.function.arguments["replace"] == .bool(true))
        #expect(
            call.function.arguments["rows"]
                == .array([
                    .object([
                        "id": .int(22),
                        "label": .string("colon-style"),
                    ])
                ]))
    }

    @Test("Nested strings keep commas, equals signs, quotes, and multiple rows")
    func nestedStringsAndMultipleRowsRemainIntact() throws {
        let output =
            #"<|tool_call_start|>[db_insert(rows=[{'id': 31, 'label': 'a,b=c'}, {'id': 32, 'label': 'Eric\'s "row"'}], table='proof_complex', replace=True)]<|tool_call_end|>"#

        let call = try #require(
            LFM2ToolCallParser().parse(content: output, tools: databaseTools()))

        #expect(call.function.arguments["replace"] == .bool(true))
        #expect(
            call.function.arguments["rows"]
                == .array([
                    .object([
                        "id": .int(31),
                        "label": .string("a,b=c"),
                    ]),
                    .object([
                        "id": .int(32),
                        "label": .string(#"Eric's "row""#),
                    ]),
                ]))
    }

    @Test("Unbalanced nested arguments never become a structured call")
    func unbalancedNestedArgumentsAreRejected() {
        let output =
            #"<|tool_call_start|>[db_insert(rows=[{'id': 11, 'label': 'broken'}], table='proof_lfm')<|tool_call_end|>"#

        #expect(LFM2ToolCallParser().parse(content: output, tools: databaseTools()) == nil)
    }

    @Test("Malformed mixed keyword fields do not get silently discarded")
    func malformedMixedKeywordFieldsAreRejected() {
        let output =
            #"<|tool_call_start|>[db_insert(rows=[{'id': 41}], invented positional junk, table='proof_lfm')]<|tool_call_end|>"#

        #expect(LFM2ToolCallParser().parse(content: output, tools: databaseTools()) == nil)
    }

    @Test("Malformed arguments cannot become an empty call for a no-argument tool")
    func malformedArgumentsDoNotMasqueradeAsEmptyCall() {
        let tools: [ToolSpec] = [
            [
                "type": "function",
                "function": [
                    "name": "ping",
                    "parameters": [
                        "type": "object",
                        "properties": [:] as [String: any Sendable],
                        "required": [] as [String],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]

        #expect(
            PythonicToolCallParser().parse(
                content: "ping(invented positional junk)",
                tools: tools) == nil)
    }

    @Test("Balanced multi-call parsing preserves nested punctuation in every call")
    func balancedMultipleCallsPreserveNestedValues() throws {
        let output =
            #"<|tool_call_start|>[db_insert(rows=[{'id': 51, 'label': '(first), still first'}], table='proof_one'), db_insert(rows=[{'id': 52, 'label': 'second) value'}], table='proof_two')]<|tool_call_end|>"#

        let calls = LFM2ToolCallParser().parseEOS(output, tools: databaseTools())

        #expect(calls.count == 2)
        let first = try #require(calls.first)
        let second = try #require(calls.last)
        #expect(first.function.arguments["table"] == .string("proof_one"))
        #expect(
            first.function.arguments["rows"]
                == .array([
                    .object([
                        "id": .int(51),
                        "label": .string("(first), still first"),
                    ])
                ]))
        #expect(second.function.arguments["table"] == .string("proof_two"))
        #expect(
            second.function.arguments["rows"]
                == .array([
                    .object([
                        "id": .int(52),
                        "label": .string("second) value"),
                    ])
                ]))
    }

    @Test("LFM nested call is stable at every legal stream boundary")
    func lfmNestedCallSurvivesEveryStreamBoundary() throws {
        let output =
            #"<|tool_call_start|>[db_insert(rows=[{'id': 61, 'label': 'Привет, (stream)'}], table='proof_stream', replace=True)]<|tool_call_end|>"#
        var boundaries = Array(output.indices.dropFirst())
        boundaries.append(output.endIndex)

        for boundary in boundaries {
            let processor = ToolCallProcessor(
                format: .lfm2,
                tools: databaseTools())
            var visible = ""
            visible += processor.processChunk(String(output[..<boundary])) ?? ""
            visible += processor.processChunk(String(output[boundary...])) ?? ""
            visible += processor.processEOS() ?? ""

            #expect(visible.isEmpty)
            #expect(processor.toolCalls.count == 1)
            let call = try #require(processor.toolCalls.first)
            #expect(call.function.name == "db_insert")
            #expect(call.function.arguments["table"] == .string("proof_stream"))
            #expect(call.function.arguments["replace"] == .bool(true))
            #expect(
                call.function.arguments["rows"]
                    == .array([
                        .object([
                            "id": .int(61),
                            "label": .string("Привет, (stream)"),
                        ])
                    ]))
        }
    }

    @Test("Malformed turn does not poison a corrected retry turn")
    func malformedThenCorrectedTurnRecovers() throws {
        let malformed =
            #"<|tool_call_start|>[db_insert(rows=[{'id': 71}], junk, table='proof_retry')]<|tool_call_end|>"#
        let corrected =
            #"<|tool_call_start|>[db_insert(rows=[{'id': 71, 'label': 'recovered'}], table='proof_retry')]<|tool_call_end|>"#

        let failedTurn = ToolCallProcessor(
            format: .lfm2,
            tools: databaseTools())
        _ = failedTurn.processChunk(malformed)
        _ = failedTurn.processEOS()
        #expect(failedTurn.toolCalls.isEmpty)

        let retryTurn = ToolCallProcessor(
            format: .lfm2,
            tools: databaseTools())
        let visible =
            (retryTurn.processChunk(corrected) ?? "")
            + (retryTurn.processEOS() ?? "")
        #expect(visible.isEmpty)
        #expect(retryTurn.toolCalls.count == 1)
        let call = try #require(retryTurn.toolCalls.first)
        #expect(call.function.arguments["table"] == .string("proof_retry"))
        #expect(
            call.function.arguments["rows"]
                == .array([
                    .object([
                        "id": .int(71),
                        "label": .string("recovered"),
                    ])
                ]))
    }

    @Test("Ambiguous inline probe keeps ordinary whitespace and prose byte-exact")
    func ambiguousInlineProbePreservesWhitespace() {
        let processor = ToolCallProcessor(
            format: .lfm2,
            tools: databaseTools())

        let visible = processor.processChunk(" \n{") ?? ""
        let eosVisible = processor.processEOS() ?? ""

        #expect(processor.toolCalls.isEmpty)
        #expect(visible + eosVisible == " \n{")
    }
}
