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

    private func liveDatabaseTools() -> [ToolSpec] {
        [
            [
                "type": "function",
                "function": [
                    "name": "db_query",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "sql": ["type": "string"] as [String: any Sendable]
                        ] as [String: any Sendable],
                        "required": ["sql"] as [String],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec,
            [
                "type": "function",
                "function": [
                    "name": "db_create_table",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "name": ["type": "string"] as [String: any Sendable],
                            "purpose": ["type": "string"] as [String: any Sendable],
                            "columns": [
                                "type": "array",
                                "items": ["type": "object"] as [String: any Sendable],
                            ] as [String: any Sendable],
                            "indexes": [
                                "type": "array",
                                "items": ["type": "object"] as [String: any Sendable],
                            ] as [String: any Sendable],
                        ] as [String: any Sendable],
                        "required": ["name", "purpose", "columns"] as [String],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec,
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

    @Test("Explicitly terminated LFM envelope missing only its list closer is quarantined")
    func explicitlyTerminatedMissingListCloserIsQuarantined() throws {
        let output =
            #"<|tool_call_start|>[db_create_table(purpose="Store parser results", name="lfm_parser_probe_three", columns=[{'name': 'label', 'type': 'TEXT'}, {'name': 'value', 'type': 'INTEGER'}])<|tool_call_end|>"#
        var boundaries = Array(output.indices.dropFirst())
        boundaries.append(output.endIndex)

        for boundary in boundaries {
            let processor = ToolCallProcessor(format: .lfm2, tools: liveDatabaseTools())
            let visible =
                (processor.processChunk(String(output[..<boundary])) ?? "")
                + (processor.processChunk(String(output[boundary...])) ?? "")
                + (processor.processEOS() ?? "")

            #expect(visible.isEmpty)
            #expect(processor.toolCalls.count == 1)
            let call = try #require(processor.toolCalls.first)
            #expect(call.function.name == "db_create_table")
            #expect(call.function.arguments["_error"] == .string("invalid_tool_arguments"))
            #expect(call.function.arguments["_tool"] == .string("db_create_table"))
            #expect(
                call.function.arguments["_message"]
                    == .string("malformed native tool envelope: missing closing ]"))
            #expect(call.function.arguments["_field"] == .string("envelope"))
            #expect(call.function.arguments["_expected"] == .string("[name(...)]"))
            #expect(call.function.arguments["columns"] == nil)
        }
    }

    @Test("Missing list closer without a native end tag remains non-executable")
    func missingListCloserWithoutEndTagRemainsRejected() {
        let output =
            #"<|tool_call_start|>[db_create_table(purpose="Store parser results", name="lfm_parser_probe_three", columns=[{'name': 'label', 'type': 'TEXT'}, {'name': 'value', 'type': 'INTEGER'}])"#
        let processor = ToolCallProcessor(format: .lfm2, tools: liveDatabaseTools())

        _ = processor.processChunk(output)
        _ = processor.processEOS()

        #expect(processor.toolCalls.isEmpty)
    }

    @Test("Unknown tool missing its list closer remains non-executable")
    func unknownToolMissingListCloserRemainsRejected() {
        let output =
            #"<|tool_call_start|>[invented_database_tool(name='x')<|tool_call_end|>"#
        let processor = ToolCallProcessor(format: .lfm2, tools: liveDatabaseTools())

        _ = processor.processChunk(output)
        _ = processor.processEOS()

        #expect(processor.toolCalls.isEmpty)
    }

    @Test("Tagged malformed mixed fields become a retryable invalid call")
    func taggedMalformedMixedKeywordFieldsAreQuarantined() throws {
        let output =
            #"<|tool_call_start|>[db_insert(rows=[{'id': 41}], invented positional junk, table='proof_lfm')]<|tool_call_end|>"#

        let call = try #require(
            LFM2ToolCallParser().parse(content: output, tools: databaseTools()))
        #expect(call.function.name == "db_insert")
        #expect(call.function.arguments["_error"] == .string("invalid_tool_arguments"))
        #expect(call.function.arguments["_field"] == .string("arguments"))
        #expect(call.function.arguments["_expected"] == .string("keyword arguments"))
        #expect(call.function.arguments["rows"] == nil)
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

    @Test("Malformed tagged turn is quarantined and does not poison a corrected retry")
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
        #expect(failedTurn.toolCalls.count == 1)
        #expect(
            failedTurn.toolCalls.first?.function.arguments["_error"]
                == .string("invalid_tool_arguments"))

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

    @Test("LFM preserves quoted SQL from the live database scenario at every stream boundary")
    func lfmPreservesQuotedSQLAtEveryStreamBoundary() throws {
        let sql =
            "SELECT name, sql, sql_version FROM sqlite_master "
            + "WHERE type='table' AND name='lfm_parser_probe' LIMIT 1;"
        let output =
            "<|tool_call_start|>[db_query(sql=\"\(sql)\")]<|tool_call_end|>"
        var boundaries = Array(output.indices.dropFirst())
        boundaries.append(output.endIndex)

        for boundary in boundaries {
            let processor = ToolCallProcessor(format: .lfm2, tools: liveDatabaseTools())
            let visible =
                (processor.processChunk(String(output[..<boundary])) ?? "")
                + (processor.processChunk(String(output[boundary...])) ?? "")
                + (processor.processEOS() ?? "")

            #expect(visible.isEmpty)
            #expect(processor.toolCalls.count == 1)
            let call = try #require(processor.toolCalls.first)
            #expect(call.function.name == "db_query")
            #expect(call.function.arguments["sql"] == .string(sql))
        }
    }

    @Test("Independent LFM database calls cannot inherit fields from a prior turn")
    func independentLFMDatabaseCallsDoNotCrossContaminate() throws {
        let query =
            #"<|tool_call_start|>[db_query(sql="SELECT name, sql, sql_version FROM sqlite_master WHERE type='table' AND name='lfm_parser_probe' LIMIT 1;")]<|tool_call_end|>"#
        let create =
            #"<|tool_call_start|>[db_create_table(name='lfm_parser_probe', purpose='Stores live parser verification values.', columns=[{'name': 'label', 'type': 'TEXT', 'nullable': False}, {'name': 'value', 'type': 'INTEGER', 'nullable': False}], indexes=[{'name': 'lfm_parser_probe_label_uq', 'columns': ['label'], 'unique': True}])]<|tool_call_end|>"#

        let queryProcessor = ToolCallProcessor(format: .lfm2, tools: liveDatabaseTools())
        _ = queryProcessor.processChunk(query)
        _ = queryProcessor.processEOS()
        let queryCall = try #require(queryProcessor.toolCalls.first)
        #expect(queryCall.function.arguments.keys.sorted() == ["sql"])

        let createProcessor = ToolCallProcessor(format: .lfm2, tools: liveDatabaseTools())
        _ = createProcessor.processChunk(create)
        _ = createProcessor.processEOS()
        let createCall = try #require(createProcessor.toolCalls.first)
        #expect(createCall.function.name == "db_create_table")
        #expect(createCall.function.arguments.keys.sorted() == ["columns", "indexes", "name", "purpose"])
        #expect(createCall.function.arguments["name"] == .string("lfm_parser_probe"))
        #expect(
            createCall.function.arguments["columns"]
                == .array([
                    .object([
                        "name": .string("label"),
                        "nullable": .bool(false),
                        "type": .string("TEXT"),
                    ]),
                    .object([
                        "name": .string("value"),
                        "nullable": .bool(false),
                        "type": .string("INTEGER"),
                    ]),
                ]))
    }

    @Test("Live duplicate columns call is rejected without choosing either value")
    func liveDuplicateColumnsCallIsQuarantined() throws {
        let output =
            #"<|tool_call_start|>[db_create_table(name='lfm_parser_probe_two', purpose="Stores live parser verification values", columns=[{'name':'label','type':'TEXT','primary_key':False},{'name':'value','type':'INTEGER','primary_key':False}], columns=[{'name':'label','type':'TEXT'},{'name':'value','type':'INTEGER'}])]<|tool_call_end|>"#

        let call = try #require(
            LFM2ToolCallParser().parse(content: output, tools: liveDatabaseTools()))

        #expect(call.function.name == "db_create_table")
        #expect(call.function.arguments["_error"] == .string("invalid_tool_arguments"))
        #expect(call.function.arguments["_tool"] == .string("db_create_table"))
        #expect(call.function.arguments["_message"] == .string("duplicate argument: columns"))
        #expect(call.function.arguments["_field"] == .string("columns"))
        #expect(
            call.function.arguments["_expected"]
                == .string("one value per declared parameter"))
        #expect(call.function.arguments["columns"] == nil)
    }

    @Test("Live duplicate columns call is quarantined at every stream boundary")
    func liveDuplicateColumnsCallIsStableAtEveryStreamBoundary() throws {
        let output =
            #"<|tool_call_start|>[db_create_table(name='lfm_parser_probe_two', purpose="Stores live parser verification values", columns=[{'name':'label','type':'TEXT'}], columns=[{'name':'value','type':'INTEGER'}])]<|tool_call_end|>"#
        var boundaries = Array(output.indices.dropFirst())
        boundaries.append(output.endIndex)

        for boundary in boundaries {
            let processor = ToolCallProcessor(format: .lfm2, tools: liveDatabaseTools())
            let visible =
                (processor.processChunk(String(output[..<boundary])) ?? "")
                + (processor.processChunk(String(output[boundary...])) ?? "")
                + (processor.processEOS() ?? "")

            #expect(visible.isEmpty)
            #expect(processor.toolCalls.count == 1)
            let call = try #require(processor.toolCalls.first)
            #expect(call.function.name == "db_create_table")
            #expect(call.function.arguments["_error"] == .string("invalid_tool_arguments"))
            #expect(call.function.arguments["_field"] == .string("columns"))
            #expect(call.function.arguments["columns"] == nil)
        }
    }

    @Test("Unknown tagged functions remain non-executable when malformed")
    func unknownTaggedMalformedFunctionRemainsRejected() {
        let output =
            #"<|tool_call_start|>[invented_database_tool(name='x', name='y')]<|tool_call_end|>"#

        #expect(LFM2ToolCallParser().parse(content: output, tools: liveDatabaseTools()) == nil)
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
