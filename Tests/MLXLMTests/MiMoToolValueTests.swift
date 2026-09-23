import Testing
@testable import MLXLMCommon

struct MiMoToolValueTests {
    private var tools: [[String: any Sendable]] {
        [["type": "function", "function": [
            "name": "file_write", "parameters": [
                "type": "object", "properties": ["content": ["type": "string"]],
                "required": ["content"],
            ] as [String: any Sendable],
        ] as [String: any Sendable]]]
    }

    // The installed native template emits strings directly inside parameter
    // tags, without adding framing newlines or JSON-escaping their contents.
    @Test(arguments: ["\n", "\r\n", "env: prod\nregion: eu-west-1\n",
                      "\n\n# Résumé — naïve café\n\n", #"literal\nnot-a-newline"#])
    func nativeStringBytesSurvive(_ value: String) throws {
        let format = try #require(ToolCallFormat.fromCapabilityName("mimo"))
        let wire = "<tool_call><function=file_write><parameter=content>\(value)</parameter></function></tool_call>"
        let call = try #require(format.createParser().parse(content: wire, tools: tools))
        #expect(call.function.arguments["content"] == .string(value))
        let processor = ToolCallProcessor(format: format, tools: tools)
        for character in wire { _ = processor.processChunk(String(character)) }
        _ = processor.processEOS()
        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.arguments["content"] == .string(value))
    }

    @Test func legacyGenericStampResolvesNativeDialect() throws {
        let result = ParserResolution.toolCall(
            capabilities: JangCapabilities(toolParser: "xml_function", supportsTools: true),
            modelType: "mimo_v2")
        let format = try #require(result.format)
        let call = try #require(format.createParser().parse(
            content: "<function=file_write><parameter=content>\n</parameter></function>", tools: tools))
        #expect(call.function.arguments["content"] == .string("\n"))
    }

    @Test func dialectResolutionPreservesOtherFamiliesAndExplicitFormats() {
        for model in ["mimo_v2", "mimo_v2_flash", "MiMo-V2.6-Flash"] {
            #expect(ToolCallFormat.infer(from: model) == .mimo)
            #expect(ToolCallFormat.fromCapabilityName("xml_function", modelType: model) == .mimo)
            #expect(ToolCallFormat.fromCapabilityName("json", modelType: model) == .json)
        }
        for stamp in ["mimo", "mimo_v2", "mimo_v2_flash", "MiMo-V2.6-Flash"] {
            #expect(ToolCallFormat.fromCapabilityName(stamp) == .mimo)
        }
        #expect(ToolCallFormat.fromCapabilityName("xml_function") == .xmlFunction)
        #expect(ToolCallFormat.fromCapabilityName("xml_function", modelType: "qwen3_6") == .xmlFunction)
    }

    @Test func qwenFramingStillTrims() throws {
        let call = try #require(ToolCallFormat.xmlFunction.createParser().parse(
            content: "<function=file_write><parameter=content>\nbody\n</parameter></function>", tools: tools))
        #expect(call.function.arguments["content"] == .string("body"))
    }

    @Test func jsonFallbackPreservesDecodedString() throws {
        let call = try #require(ToolCallFormat.mimo.createParser().parse(
            content: #"<tool_call>{"name":"file_write","arguments":{"content":"\n"}}</tool_call>"#,
            tools: tools))
        #expect(call.function.arguments["content"] == .string("\n"))
    }

    @Test func typedParametersRetainConversion() throws {
        let schema: [[String: any Sendable]] = [["type": "function", "function": [
            "name": "typed", "parameters": ["type": "object", "properties": [
                "count": ["type": "integer"], "enabled": ["type": "boolean"],
                "data": ["type": "object"],
            ]] as [String: any Sendable],
        ] as [String: any Sendable]]]
        let call = try #require(ToolCallFormat.mimo.createParser().parse(
            content: "<function=typed><parameter=count>\n2\n</parameter>"
                + "<parameter=enabled>\ntrue\n</parameter>"
                + "<parameter=data>\n{\"text\":\"line\\n\"}\n</parameter></function>", tools: schema))
        #expect(call.function.arguments["count"] == .int(2))
        #expect(call.function.arguments["enabled"] == .bool(true))
        #expect(call.function.arguments["data"] == .object(["text": .string("line\n")]))
    }
}
