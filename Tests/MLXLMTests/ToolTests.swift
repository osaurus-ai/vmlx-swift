import Foundation
import MLXLMCommon
import Testing

struct ToolTests {
    private func lineCountTools() -> [ToolSpec] {
        [
            [
                "type": "function",
                "function": [
                    "name": "line_count",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "text": ["type": "string"] as [String: any Sendable],
                        ] as [String: any Sendable],
                        "required": ["text"] as [String],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]
    }

    @Test("Test Weather Tool Schema Generation")
    func testWeatherToolSchemaGeneration() throws {
        struct WeatherInput: Codable {
            let location: String
            let unit: String?
        }

        struct WeatherOutput: Codable {
            let temperature: Double
            let conditions: String
        }

        let tool = Tool<WeatherInput, WeatherOutput>(
            name: "get_current_weather",
            description: "Get the current weather in a given location",
            parameters: [
                .required(
                    "location", type: .string, description: "The city, e.g. Istanbul"
                ),
                .optional(
                    "unit",
                    type: .string,
                    description: "The unit of temperature",
                    extraProperties: [
                        "enum": ["celsius", "fahrenheit"]
                    ]
                ),
            ]
        ) { input in
            WeatherOutput(temperature: 14.0, conditions: "Sunny")
        }

        let actual = tool.schema as NSDictionary

        let expected: NSDictionary = [
            "type": "function",
            "function": [
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "location": [
                            "type": "string",
                            "description": "The city, e.g. Istanbul",
                        ],
                        "unit": [
                            "type": "string",
                            "description": "The unit of temperature",
                            "enum": ["celsius", "fahrenheit"],
                        ],
                    ],
                    "required": ["location"],
                ],
            ],
        ]

        #expect(actual == expected)
    }

    @Test("Test Tool Call Detection in Generated Text - Default JSON Format")
    func testToolCallDetection() throws {
        let processor = ToolCallProcessor()
        let chunks: [String] = [
            "<tool", "_", "call>", "{", "\"", "name", "\"", ":", " ", "\"", "get", "_", "current",
            "_", "weather", "\"", ",", " ", "\"", "arguments", "\"", ":", " ", "{", "\"",
            "location", "\"", ":", " ", "\"", "San", " Francisco", "\"", ",", " ", "\"", "unit",
            "\"", ":", " ", "\"", "celsius", "\"", "}", "}", "</tool", "_", "call>",
        ]

        for chunk in chunks {
            let result = processor.processChunk(chunk)
            #expect(result == nil)
        }

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)

        #expect(toolCall.function.name == "get_current_weather")
        #expect(toolCall.function.arguments["location"] == .string("San Francisco"))
        #expect(toolCall.function.arguments["unit"] == .string("celsius"))
    }

    // MARK: - JSON Format Tests

    @Test("Test JSON Tool Call Parser - Default Tags")
    func testJSONParserDefaultTags() throws {
        let parser = JSONToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let content =
            "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}</tool_call>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Paris"))
    }

    @Test("Test JSON Tool Call Parser - Custom Tags")
    func testJSONParserCustomTags() throws {
        let parser = JSONToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let content =
            "<|tool_call_start|>{\"name\": \"search\", \"arguments\": {\"query\": \"swift programming\"}}<|tool_call_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "search")
        #expect(toolCall.function.arguments["query"] == .string("swift programming"))
    }

    // MARK: - Pythonic Format Tests (LFM2/LFM2.5)

    @Test("Test Pythonic Tool Call Parser - Basic")
    func testPythonicParserBasic() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let content =
            "<|tool_call_start|>[get_weather(location='Paris', unit='celsius')]<|tool_call_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Paris"))
        #expect(toolCall.function.arguments["unit"] == .string("celsius"))
    }

    @Test("Test Pythonic Tool Call Parser - Double Quotes")
    func testPythonicParserDoubleQuotes() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let content =
            "<|tool_call_start|>[search(query=\"swift programming\")]<|tool_call_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "search")
        #expect(toolCall.function.arguments["query"] == .string("swift programming"))
    }

    @Test("Test Pythonic Tool Call Parser - Without Brackets")
    func testPythonicParserWithoutBrackets() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let content =
            "<|tool_call_start|>current_time(timezone='UTC')<|tool_call_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "current_time")
        #expect(toolCall.function.arguments["timezone"] == .string("UTC"))
    }

    @Test("Test Pythonic Tool Call Parser - Nested Parentheses in Argument Value")
    func testPythonicParserNestedParentheses() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let content =
            "<|tool_call_start|>[run_script(code=\"response = requests.get('https://api.example.com/data')\")] <|tool_call_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "run_script")
        #expect(
            toolCall.function.arguments["code"]
                == .string("response = requests.get('https://api.example.com/data')"))
    }

    @Test("Test Pythonic Tool Call Parser - Nested Parentheses Without Brackets")
    func testPythonicParserNestedParenthesesNoBrackets() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let content =
            "<|tool_call_start|>run_script(code=\"print('hello')\")<|tool_call_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "run_script")
        #expect(toolCall.function.arguments["code"] == .string("print('hello')"))
    }

    @Test("Test Pythonic Tool Call Parser - No Arguments")
    func testPythonicParserNoArguments() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let content =
            "<|tool_call_start|>[current_time()]<|tool_call_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "current_time")
        #expect(toolCall.function.arguments.isEmpty)
    }

    @Test("Test Pythonic Tool Call Parser - Type Conversion")
    func testPythonicParserTypeConversion() throws {
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let tools: [[String: any Sendable]] = [
            [
                "function": [
                    "name": "set_temperature",
                    "parameters": [
                        "properties": [
                            "value": ["type": "integer"],
                            "enabled": ["type": "boolean"],
                        ]
                    ],
                ] as [String: any Sendable]
            ]
        ]
        let content =
            "<|tool_call_start|>[set_temperature(value='25', enabled='true')]<|tool_call_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: tools))

        #expect(toolCall.function.name == "set_temperature")
        #expect(toolCall.function.arguments["value"] == .int(25))
        #expect(toolCall.function.arguments["enabled"] == .bool(true))
    }

    @Test("Test LFM2 Format via ToolCallProcessor - Pythonic")
    func testLFM2FormatProcessor() throws {
        let processor = ToolCallProcessor(format: .lfm2)
        let content =
            "<|tool_call_start|>[calculator(expression='2+2')]<|tool_call_end|>"

        _ = processor.processChunk(content)

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "calculator")
        #expect(toolCall.function.arguments["expression"] == .string("2+2"))
    }

    @Test("LFM2 processor accepts bare native call list")
    func lfm2ProcessorAcceptsBareNativeCallList() throws {
        let tools: [ToolSpec] = [
            [
                "type": "function",
                "function": [
                    "name": "line_count",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "text": ["type": "string"] as [String: any Sendable],
                        ] as [String: any Sendable],
                        "required": ["text"] as [String],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]
        let processor = ToolCallProcessor(format: .lfm2, tools: tools)
        let visible = processor.processChunk(#"[line_count(text="red\ngreen\nblue")]"#)
        _ = processor.processEOS()

        #expect(visible == nil)
        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "line_count")
        #expect(toolCall.function.arguments["text"] == .string("red\ngreen\nblue"))
    }

    @Test("LFM2 parser maps one positional string to the required parameter")
    func lfm2ParserMapsOnePositionalStringToRequiredParameter() throws {
        let tools: [ToolSpec] = [
            [
                "type": "function",
                "function": [
                    "name": "line_count",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "text": ["type": "string"] as [String: any Sendable],
                        ] as [String: any Sendable],
                        "required": ["text"] as [String],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let toolCall = try #require(
            parser.parse(content: #"<|tool_call_start|>[line_count("red\ngreen\nblue")]<|tool_call_end|>"#, tools: tools))

        #expect(toolCall.function.name == "line_count")
        #expect(toolCall.function.arguments["text"] == .string("red\ngreen\nblue"))
    }

    @Test("LFM2 parser decodes escaped newlines in native strings")
    func lfm2ParserDecodesEscapedNewlinesInNativeStrings() throws {
        let tools: [ToolSpec] = [
            [
                "type": "function",
                "function": [
                    "name": "line_count",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "text": ["type": "string"] as [String: any Sendable],
                        ] as [String: any Sendable],
                        "required": ["text"] as [String],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let toolCall = try #require(
            parser.parse(content: #"<|tool_call_start|>[line_count(text="red\ngreen\nblue")]<|tool_call_end|>"#, tools: tools))

        #expect(toolCall.function.name == "line_count")
        #expect(toolCall.function.arguments["text"] == .string("red\ngreen\nblue"))
    }

    @Test("LFM2 parser accepts observed bracketed name and JSON arguments array")
    func lfm2ParserAcceptsObservedBracketedNameAndJSONArgumentsArray() throws {
        let tools = lineCountTools()
        let parser = PythonicToolCallParser(
            startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        let toolCall = try #require(
            parser.parse(content: #"[ "line_count", { "text": "alpha\nbeta\ngamma" } ]"#, tools: tools))

        #expect(toolCall.function.name == "line_count")
        #expect(toolCall.function.arguments["text"] == .string("alpha\nbeta\ngamma"))
    }

    @Test("LFM2 processor accepts tool-keyed JSON object envelope at EOS")
    func lfm2ProcessorAcceptsToolKeyedJSONObjectEnvelopeAtEOS() throws {
        let tools: [ToolSpec] = [
            [
                "type": "function",
                "function": [
                    "name": "line_count",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "text": ["type": "string"] as [String: any Sendable],
                        ] as [String: any Sendable],
                        "required": ["text"] as [String],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]
        let processor = ToolCallProcessor(format: .lfm2, tools: tools)
        let visible = processor.processChunk(
            #"""
            {
              "line_count": [
                {
                  "text": "red\ngreen\nblue"
              }
            ]
            """#)
        let eosVisible = processor.processEOS()

        #expect(visible == nil)
        #expect(eosVisible == nil)
        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "line_count")
        #expect(toolCall.function.arguments["text"] == .string("red\ngreen\nblue"))
    }

    @Test("LFM2 parser accepts single-tool JSON argument object")
    func lfm2ParserAcceptsSingleToolJSONArgumentObject() throws {
        let tools: [ToolSpec] = [
            [
                "type": "function",
                "function": [
                    "name": "line_count",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "text": ["type": "string"] as [String: any Sendable],
                        ] as [String: any Sendable],
                        "required": ["text"] as [String],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]
        let parser = PythonicToolCallParser()
        let toolCall = try #require(
            parser.parse(content: #"{"text":"red\ngreen\nblue"}"#, tools: tools))

        #expect(toolCall.function.name == "line_count")
        #expect(toolCall.function.arguments["text"] == .string("red\ngreen\nblue"))
    }

    @Test("LFM2 parser accepts single-tool JSON arguments envelope")
    func lfm2ParserAcceptsSingleToolJSONArgumentsEnvelope() throws {
        let tools: [ToolSpec] = [
            [
                "type": "function",
                "function": [
                    "name": "line_count",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "text": ["type": "string"] as [String: any Sendable],
                        ] as [String: any Sendable],
                        "required": ["text"] as [String],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]
        let parser = PythonicToolCallParser()
        let toolCall = try #require(
            parser.parse(content: #"{"arguments":{"text":"one\ntwo"}}"#, tools: tools))

        #expect(toolCall.function.name == "line_count")
        #expect(toolCall.function.arguments["text"] == .string("one\ntwo"))
    }

    @Test("LFM2 parser rejects single-tool JSON object missing required args")
    func lfm2ParserRejectsSingleToolJSONObjectMissingRequiredArgs() throws {
        let tools: [ToolSpec] = [
            [
                "type": "function",
                "function": [
                    "name": "line_count",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "text": ["type": "string"] as [String: any Sendable],
                        ] as [String: any Sendable],
                        "required": ["text"] as [String],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]
        let parser = PythonicToolCallParser()
        let toolCall = parser.parse(content: #"{"path":"/tmp/a"}"#, tools: tools)

        #expect(toolCall == nil)
    }

    @Test("LFM2 parser ignores unknown tool-keyed JSON envelopes")
    func lfm2ParserIgnoresUnknownToolKeyedJSONEnvelope() throws {
        let tools: [ToolSpec] = [
            [
                "type": "function",
                "function": [
                    "name": "line_count",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "text": ["type": "string"] as [String: any Sendable],
                        ] as [String: any Sendable],
                        "required": ["text"] as [String],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]
        let parser = PythonicToolCallParser()
        let toolCall = parser.parse(content: #"{"not_a_tool":[{"text":"red"}]}"#, tools: tools)

        #expect(toolCall == nil)
    }

    @Test("LFM2 processor accepts observed functionline required-tool output at EOS")
    func lfm2ProcessorAcceptsObservedFunctionlineRequiredToolOutput() throws {
        let tools = lineCountTools()
        let processor = ToolCallProcessor(format: .lfm2, tools: tools)
        let visible = processor.processChunk(
            #"<functionline_count>line_count</function_line_count>text='red\ngreen\nblue'</function_line_count>"#)
        let eosVisible = processor.processEOS()

        #expect(visible == nil)
        #expect(eosVisible == nil)
        #expect(processor.toolCalls.count == 1)
        let call = try #require(processor.toolCalls.first)
        #expect(call.function.name == "line_count")
        #expect(call.function.arguments["text"] == .string("red\ngreen\nblue"))
    }

    @Test("LFM2 processor accepts observed functionname arg tag output at EOS")
    func lfm2ProcessorAcceptsObservedFunctionNameArgTagOutput() throws {
        let tools = lineCountTools()
        let parserCall = try #require(LFM2ToolCallParser().parse(
            content: #"""
            <functionname>line_count</functionname>
            <argimiss name="text">one
            two</argFunctionName>
            """#,
            tools: tools))
        #expect(parserCall.function.arguments["text"] == .string("one\ntwo"))

        let processor = ToolCallProcessor(format: .lfm2, tools: tools)
        let visible = processor.processChunk(
            #"""
            <functionname>line_count</functionname>
            <argimiss name="text">one
            two</argFunctionName>
            """#)
        let eosVisible = processor.processEOS()

        #expect(visible == nil)
        #expect(eosVisible == nil)
        #expect(processor.toolCalls.count == 1)
        let call = try #require(processor.toolCalls.first)
        #expect(call.function.name == "line_count")
        #expect(call.function.arguments["text"] == .string("one\ntwo"))
    }

    @Test("LFM2 parser rejects native pythonic calls outside supplied tool schema")
    func lfm2ParserRejectsNativePythonicCallsOutsideSuppliedToolSchema() {
        let tools = lineCountTools()
        let parser = LFM2ToolCallParser()

        #expect(parser.parse(
            content: #"<|tool_call_start|>[break()]<|tool_call_end|>"#,
            tools: tools) == nil)
        #expect(parser.parseEOS(
            #"<|tool_call_start|>[break()]<|tool_call_end|>"#,
            tools: tools).isEmpty)

        let processor = ToolCallProcessor(format: .lfm2, tools: tools)
        #expect(processor.processChunk(#"<|tool_call_start|>[break()]<|tool_call_end|>"#) == nil)
        #expect(processor.processEOS() == nil)
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("LFM2 parser does not coerce plain code fence into a tool call")
    func lfm2ParserDoesNotCoercePlainCodeFenceIntoToolCall() {
        let processor = ToolCallProcessor(format: .lfm2, tools: lineCountTools())
        let visible = processor.processChunk(
            #"""
            ```lfm
            red
            green
            blue
            ```
            """#)
        let eosVisible = processor.processEOS()

        #expect(processor.toolCalls.isEmpty)
        #expect((visible ?? "") + (eosVisible ?? "") != "")
    }

    // MARK: - XML Function Format Tests (Qwen3 Coder)

    @Test("Test XML Function Parser - Qwen3 Coder Format")
    func testXMLFunctionParser() throws {
        let parser = XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let content =
            "<function=get_weather><parameter=location>Tokyo</parameter><parameter=unit>celsius</parameter></function>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Tokyo"))
        #expect(toolCall.function.arguments["unit"] == .string("celsius"))
    }

    @Test("Test XML Function Parser - With Type Conversion")
    func testXMLFunctionParserTypeConversion() throws {
        let parser = XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let tools: [[String: any Sendable]] = [
            [
                "function": [
                    "name": "set_temperature",
                    "parameters": [
                        "properties": [
                            "value": ["type": "integer"],
                            "enabled": ["type": "boolean"],
                        ]
                    ],
                ] as [String: any Sendable]
            ]
        ]
        let content =
            "<function=set_temperature><parameter=value>25</parameter><parameter=enabled>true</parameter></function>"

        let toolCall = try #require(parser.parse(content: content, tools: tools))

        #expect(toolCall.function.name == "set_temperature")
        #expect(toolCall.function.arguments["value"] == .int(25))
        #expect(toolCall.function.arguments["enabled"] == .bool(true))
    }

    @Test("Test XML Function Parser - Multiline Content (Qwen3.5 style)")
    func testXMLFunctionParserMultiline() throws {
        let parser = XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
        // Qwen3.5 models generate newlines between the XML tags
        let content = """
            <tool_call>
            <function=get_current_datetime>
            </function>
            </tool_call>
            """

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_current_datetime")
        #expect(toolCall.function.arguments.isEmpty)
    }

    @Test("Test XML Function Parser - Multiline Parameters")
    func testXMLFunctionParserMultilineParams() throws {
        let parser = XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let content = """
            <function=get_weather>
            <parameter=location>
            Tokyo
            </parameter>
            </function>
            """

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Tokyo"))
    }

    // MARK: - Qwen3.5 Format Tests (XML Function with tool_call wrapper)

    @Test("Test Qwen3.5 XML Function Parser - With tool_call Tags")
    func testQwen35Parser() throws {
        let parser = XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let content = """
            <tool_call>
            <function=get_weather>
            <parameter=location>
            San Francisco
            </parameter>
            <parameter=unit>
            celsius
            </parameter>
            </function>
            </tool_call>
            """

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("San Francisco"))
        #expect(toolCall.function.arguments["unit"] == .string("celsius"))
    }

    @Test("Test Qwen3.5 Format via ToolCallProcessor")
    func testQwen35FormatProcessor() throws {
        let processor = ToolCallProcessor(format: .xmlFunction)
        let chunks: [String] = [
            "<tool", "_call>", "\n<function=get_weather>\n",
            "<parameter=location>\nTokyo\n</parameter>",
            "\n</function>\n</tool_call>",
        ]

        for chunk in chunks {
            _ = processor.processChunk(chunk)
        }

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Tokyo"))
    }

    @Test("Test Qwen3.5 Format - No Arguments")
    func testQwen35FormatNoArgs() throws {
        let processor = ToolCallProcessor(format: .xmlFunction)
        let content = "<tool_call>\n<function=get_current_datetime>\n</function>\n</tool_call>"

        _ = processor.processChunk(content)

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_current_datetime")
        #expect(toolCall.function.arguments.isEmpty)
    }

    @Test("XML Function Parser marks missing required arguments invalid")
    func testXMLFunctionParserMissingRequiredArgument() throws {
        let parser = XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let swiftTools: [[String: any Sendable]] = [
            [
                "function": [
                    "name": "line_count",
                    "parameters": [
                        "properties": [
                            "text": ["type": "string"] as [String: any Sendable]
                        ] as [String: any Sendable],
                        "required": ["text"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable]
            ]
        ]
        let content = "<tool_call>\n<function=line_count>\n</function>\n</tool_call>"

        let toolCall = try #require(parser.parse(content: content, tools: swiftTools))

        #expect(toolCall.function.name == "line_count")
        #expect(toolCall.function.arguments["text"] == nil)
        #expect(toolCall.function.arguments["_error"] == .string("invalid_tool_arguments"))
        #expect(toolCall.function.arguments["_field"] == .string("text"))

        let bridgedTools = try #require(
            JSONSerialization.jsonObject(
                with: JSONSerialization.data(withJSONObject: swiftTools),
                options: []) as? [[String: any Sendable]]
        )
        let bridgedToolCall = try #require(parser.parse(content: content, tools: bridgedTools))
        #expect(bridgedToolCall.function.arguments["_error"] == .string("invalid_tool_arguments"))
        #expect(bridgedToolCall.function.arguments["_field"] == .string("text"))
    }

    // MARK: - GLM4 Format Tests

    @Test("Test GLM4 Tool Call Parser")
    func testGLM4Parser() throws {
        let parser = GLM4ToolCallParser()
        let content =
            "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Berlin</arg_value><arg_key>unit</arg_key><arg_value>celsius</arg_value></tool_call>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Berlin"))
        #expect(toolCall.function.arguments["unit"] == .string("celsius"))
    }

    @Test("Test GLM4 Format via ToolCallProcessor")
    func testGLM4FormatProcessor() throws {
        let processor = ToolCallProcessor(format: .glm4)
        let content =
            "<tool_call>search<arg_key>query</arg_key><arg_value>machine learning</arg_value></tool_call>"

        _ = processor.processChunk(content)

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "search")
        #expect(toolCall.function.arguments["query"] == .string("machine learning"))
    }

    // MARK: - Gemma Format Tests

    @Test("Test Gemma Function Parser")
    func testGemmaParser() throws {
        let parser = GemmaFunctionParser()
        let content =
            "<start_function_call>call:get_weather{location:Paris,unit:celsius}<end_function_call>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Paris"))
        #expect(toolCall.function.arguments["unit"] == .string("celsius"))
    }

    @Test("Test Gemma Function Parser - Escaped Strings")
    func testGemmaParserEscapedStrings() throws {
        let parser = GemmaFunctionParser()
        // Note: Gemma uses <escape> for both start and end markers (not </escape>)
        let content =
            "<start_function_call>call:search{query:<escape>hello, world!<escape>}<end_function_call>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "search")
        #expect(toolCall.function.arguments["query"] == .string("hello, world!"))
    }

    @Test("Test Gemma Format via ToolCallProcessor")
    func testGemmaFormatProcessor() throws {
        let processor = ToolCallProcessor(format: .gemma)
        let content = "<start_function_call>call:calculator{expression:2+2}<end_function_call>"

        _ = processor.processChunk(content)

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "calculator")
        #expect(toolCall.function.arguments["expression"] == .string("2+2"))
    }

    // MARK: - Gemma 4 Format Tests

    @Test("Test Gemma 4 Function Parser - simple args")
    func testGemma4ParserSimple() throws {
        let parser = GemmaFunctionParser(
            startTag: "<|tool_call>", endTag: "<tool_call|>", escapeMarker: "<|\"|>")
        let content = "<|tool_call>call:add{a:5,b:3}<tool_call|>"
        let toolCall = try #require(parser.parse(content: content, tools: nil))
        #expect(toolCall.function.name == "add")
    }

    @Test("Test Gemma 4 Function Parser - escaped strings")
    func testGemma4ParserEscapedStrings() throws {
        let parser = GemmaFunctionParser(
            startTag: "<|tool_call>", endTag: "<tool_call|>", escapeMarker: "<|\"|>")
        let content =
            "<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>,unit:<|\"|>celsius<|\"|>}<tool_call|>"
        let toolCall = try #require(parser.parse(content: content, tools: nil))
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Paris"))
        #expect(toolCall.function.arguments["unit"] == .string("celsius"))
    }

    @Test("Test Gemma 4 Function Parser - comma inside escaped value")
    func testGemma4ParserCommaInString() throws {
        let parser = GemmaFunctionParser(
            startTag: "<|tool_call>", endTag: "<tool_call|>", escapeMarker: "<|\"|>")
        let content =
            "<|tool_call>call:search{query:<|\"|>hello, world<|\"|>}<tool_call|>"
        let toolCall = try #require(parser.parse(content: content, tools: nil))
        #expect(toolCall.function.name == "search")
        #expect(toolCall.function.arguments["query"] == .string("hello, world"))
    }

    @Test("Test Gemma 4 Format via ToolCallProcessor (auto-infer)")
    func testGemma4FormatProcessor() throws {
        // Verify the factory gemma4 case wires the correct tags and escape marker.
        let processor = ToolCallProcessor(format: .gemma4)
        _ = processor.processChunk(
            "<|tool_call>call:calculator{expression:<|\"|>2+2<|\"|>}<tool_call|>")
        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "calculator")
        #expect(toolCall.function.arguments["expression"] == .string("2+2"))
    }

    @Test("Test ToolCallFormat.infer maps gemma4 model_type correctly")
    func testGemma4FormatInference() throws {
        #expect(ToolCallFormat.infer(from: "gemma4") == .gemma4)
        #expect(ToolCallFormat.infer(from: "gemma4_text") == .gemma4)
        #expect(ToolCallFormat.infer(from: "gemma3") == .gemma)
        #expect(ToolCallFormat.infer(from: "gemma3_text") == .gemma)
        #expect(ToolCallFormat.infer(from: "gemma3n") == nil)
    }

    // MARK: - Kimi K2 Format Tests

    @Test("Test Kimi K2 Tool Call Parser")
    func testKimiK2Parser() throws {
        let parser = KimiK2ToolCallParser()
        let content =
            "<|tool_calls_section_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\": \"London\"}<|tool_calls_section_end|>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("London"))
    }

    @Test("Test Kimi K2 Format via ToolCallProcessor")
    func testKimiK2FormatProcessor() throws {
        let processor = ToolCallProcessor(format: .kimiK2)
        let content =
            "<|tool_calls_section_begin|>functions.search:0<|tool_call_argument_begin|>{\"query\": \"swift\"}<|tool_calls_section_end|>"

        _ = processor.processChunk(content)

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "search")
        #expect(toolCall.function.arguments["query"] == .string("swift"))
    }

    // MARK: - MiniMax M2 Format Tests

    @Test("Test MiniMax M2 Tool Call Parser")
    func testMiniMaxM2Parser() throws {
        let parser = MiniMaxM2ToolCallParser()
        let content =
            "<minimax:tool_call><invoke name=\"get_weather\"><parameter name=\"location\">Sydney</parameter></invoke></minimax:tool_call>"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Sydney"))
    }

    @Test("Test MiniMax M2 Format via ToolCallProcessor")
    func testMiniMaxM2FormatProcessor() throws {
        let processor = ToolCallProcessor(format: .minimaxM2)
        let content =
            "<minimax:tool_call><invoke name=\"search\"><parameter name=\"query\">AI news</parameter></invoke></minimax:tool_call>"

        _ = processor.processChunk(content)

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "search")
        #expect(toolCall.function.arguments["query"] == .string("AI news"))
    }

    // MARK: - ZAYA XML Format Tests

    @Test("Test ZAYA XML Function Parser")
    func testZayaXMLFunctionParser() throws {
        let processor = ToolCallProcessor(format: .zayaXml)
        let chunks = [
            "Before\n",
            "<zyphra_tool_call>\n<function=search_web>\n<parameter=query>\n",
            "ZAYA CCA cache\n</parameter>\n</function>\n</zyphra_tool_call>\nAfter",
        ]

        var visible = ""
        for chunk in chunks {
            if let out = processor.processChunk(chunk) {
                visible += out
            }
        }

        #expect(visible.contains("Before"))
        #expect(visible.contains("After"))
        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "search_web")
        #expect(toolCall.function.arguments["query"] == .string("ZAYA CCA cache"))
    }

    @Test("ZAYA XML parser trims protocol-edge newlines from string parameters")
    func testZayaXMLParserTrimsProtocolEdgeNewlines() throws {
        let parameters: [String: any Sendable] = [
            "type": "object",
            "properties": [
                "text": ["type": "string"] as [String: any Sendable],
            ] as [String: any Sendable],
            "required": ["text"],
        ]
        let function: [String: any Sendable] = [
            "name": "line_count",
            "parameters": parameters,
        ]
        let processor = ToolCallProcessor(format: .zayaXml, tools: [
            [
                "type": "function",
                "function": function,
            ] as [String: any Sendable]
        ])
        let content = """
            <zyphra_tool_call>
            <function=line_count>
            <parameter=text>
            one
            two

            </parameter>
            </function>
            </zyphra_tool_call>
            """

        _ = processor.processChunk(content)

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "line_count")
        #expect(toolCall.function.arguments["text"] == .string("one\ntwo"))
    }

    // MARK: - ToolCallFormat Serialization Tests

    @Test("Test ToolCallFormat Raw Values for Serialization")
    func testToolCallFormatRawValues() throws {
        // Test that raw values are suitable for JSON/CLI serialization
        #expect(ToolCallFormat.json.rawValue == "json")
        #expect(ToolCallFormat.lfm2.rawValue == "lfm2")
        #expect(ToolCallFormat.xmlFunction.rawValue == "xml_function")
        #expect(ToolCallFormat.glm4.rawValue == "glm4")
        #expect(ToolCallFormat.gemma.rawValue == "gemma")
        #expect(ToolCallFormat.kimiK2.rawValue == "kimi_k2")
        #expect(ToolCallFormat.minimaxM2.rawValue == "minimax_m2")
        #expect(ToolCallFormat.mistral.rawValue == "mistral")
        #expect(ToolCallFormat.zayaXml.rawValue == "zaya_xml")

        // Test round-trip via raw value
        for format in ToolCallFormat.allCases {
            #expect(ToolCallFormat(rawValue: format.rawValue) == format)
        }
    }

    // MARK: - Format Inference Tests

    @Test("Test ToolCallFormat Inference from Model Type")
    func testToolCallFormatInference() throws {
        // LFM2 models (prefix matching)
        #expect(ToolCallFormat.infer(from: "lfm2") == .lfm2)
        #expect(ToolCallFormat.infer(from: "LFM2") == .lfm2)
        #expect(ToolCallFormat.infer(from: "lfm2_moe") == .lfm2)
        #expect(ToolCallFormat.infer(from: "lfm2_5") == .lfm2)
        #expect(ToolCallFormat.infer(from: "LFM2_5") == .lfm2)
        #expect(ToolCallFormat.infer(from: "lfm25") == .lfm2)

        // GLM4 models (prefix matching)
        #expect(ToolCallFormat.infer(from: "glm4") == .glm4)
        #expect(ToolCallFormat.infer(from: "glm4_moe") == .glm4)
        #expect(ToolCallFormat.infer(from: "glm4_moe_lite") == .glm4)
        #expect(ToolCallFormat.infer(from: "glm4_5") == .glm4)
        #expect(ToolCallFormat.infer(from: "GLM4_5") == .glm4)

        // Gemma models
        #expect(ToolCallFormat.infer(from: "gemma") == .gemma)
        #expect(ToolCallFormat.infer(from: "GEMMA") == .gemma)

        // Nemotron models (prefix matching)
        #expect(ToolCallFormat.infer(from: "nemotron_h") == .nemotron)
        #expect(ToolCallFormat.infer(from: "NEMOTRON_H") == .nemotron)

        // Qwen3.5 models (prefix matching)
        #expect(ToolCallFormat.infer(from: "qwen3_5") == .xmlFunction)
        #expect(ToolCallFormat.infer(from: "qwen3_5_moe") == .xmlFunction)
        #expect(ToolCallFormat.infer(from: "QWEN3_5") == .xmlFunction)

        // Mistral3 models (prefix matching)
        #expect(ToolCallFormat.infer(from: "mistral3") == .mistral)
        #expect(ToolCallFormat.infer(from: "Mistral3") == .mistral)
        #expect(ToolCallFormat.infer(from: "mistral3_text") == .mistral)

        // ZAYA / Zyphra models (prefix matching)
        #expect(ToolCallFormat.infer(from: "zaya") == .zayaXml)
        #expect(ToolCallFormat.infer(from: "ZAYA") == .zayaXml)
        #expect(ToolCallFormat.infer(from: "zaya_text") == .zayaXml)
        #expect(ToolCallFormat.infer(from: "zyphra_zaya") == .zayaXml)

        // Unknown models should return nil (use default JSON format)
        #expect(ToolCallFormat.infer(from: "llama") == nil)
        #expect(ToolCallFormat.infer(from: "qwen2") == nil)
        #expect(ToolCallFormat.infer(from: "mistral") == nil)
    }

    // MARK: - Mistral Format Tests

    @Test("Test Mistral Tool Call Parser")
    func testMistralParser() throws {
        let parser = MistralToolCallParser()
        let content = "[TOOL_CALLS]get_weather [ARGS]{\"location\": \"Paris\"}"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Paris"))
    }

    @Test("Test Mistral Tool Call Parser - With Call ID")
    func testMistralParserWithCallId() throws {
        let parser = MistralToolCallParser()
        let content = "[TOOL_CALLS]get_weather[CALL_ID]abc123xyz[ARGS]{\"location\": \"Paris\"}"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Paris"))
    }

    @Test("Test Mistral Tool Call Parser - Preserves [TOOL_CALLS] in Arguments")
    func testMistralParserPreservesStartTagInArguments() throws {
        let parser = MistralToolCallParser()
        let content = "get_note[ARGS]{\"text\": \"literal [TOOL_CALLS] marker\"}"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_note")
        #expect(toolCall.function.arguments["text"] == .string("literal [TOOL_CALLS] marker"))
    }

    @Test("Test Mistral Tool Call Parser - Preserves </s> in Arguments")
    func testMistralParserPreservesEndTagInArguments() throws {
        let parser = MistralToolCallParser()
        let content = "get_note[ARGS]{\"text\": \"literal </s> marker\"}"

        let toolCall = try #require(parser.parse(content: content, tools: nil))

        #expect(toolCall.function.name == "get_note")
        #expect(toolCall.function.arguments["text"] == .string("literal </s> marker"))
    }

    @Test("Test Mistral Format via ToolCallProcessor")
    func testMistralFormatProcessor() throws {
        let processor = ToolCallProcessor(format: .mistral)
        let chunks: [String] = [
            "[TOOL", "_CALLS]", "get_weather", " [ARGS]",
            "{\"location\":", " \"Tokyo\"}",
        ]

        for chunk in chunks {
            _ = processor.processChunk(chunk)
        }

        // End tag never arrives in text, so tool call stays buffered until processEOS
        #expect(processor.toolCalls.count == 0)

        processor.processEOS()

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Tokyo"))
    }

    @Test("Test Mistral Format Processor EOS")
    func testMistralFormatProcessorEOS() throws {
        let processor = ToolCallProcessor(format: .mistral)
        let content = "[TOOL_CALLS]get_weather [ARGS]{\"location\": \"Berlin\"}"

        _ = processor.processChunk(content)

        // Before processEOS, no tool calls extracted (end tag never arrives)
        #expect(processor.toolCalls.count == 0)

        // processEOS extracts the buffered tool call
        processor.processEOS()

        #expect(processor.toolCalls.count == 1)
        let toolCall = try #require(processor.toolCalls.first)
        #expect(toolCall.function.name == "get_weather")
        #expect(toolCall.function.arguments["location"] == .string("Berlin"))
    }

    @Test("Test Mistral Format Processor Multiple Tool Calls")
    func testMistralFormatProcessorMultipleToolCalls() throws {
        let processor = ToolCallProcessor(format: .mistral)
        let chunks: [String] = [
            "[TOOL_CALLS]get_weather[ARGS]",
            "{\"location\": \"Paris\"}",
            "[TOOL_CALLS]get_time",
            "[ARGS]{\"timezone\": \"UTC\"}",
        ]

        for chunk in chunks {
            let result = processor.processChunk(chunk)
            // All chunks should be buffered (nil) after the start tag
            if chunk == chunks.first {
                #expect(result == nil)
            }
        }

        // No tool calls before processEOS
        #expect(processor.toolCalls.count == 0)

        processor.processEOS()

        // Both tool calls should be extracted
        #expect(processor.toolCalls.count == 2)

        let first = try #require(processor.toolCalls.first)
        #expect(first.function.name == "get_weather")
        #expect(first.function.arguments["location"] == .string("Paris"))

        let second = processor.toolCalls[1]
        #expect(second.function.name == "get_time")
        #expect(second.function.arguments["timezone"] == .string("UTC"))
    }
}
