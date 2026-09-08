import Foundation
import Testing
@testable import MLXLMCommon

@Suite("MiniCPM5 native XML tool contract — no model allocation")
struct MiniCPM5ToolCallParserTests {
    private let parser = MiniCPM5ToolCallParser()
    private static let tools: [[String: any Sendable]] = {
        let data = Data("""
            [{
              "type": "function",
              "function": {
                "name": "f",
                "parameters": {
                  "type": "object",
                  "properties": {
                    "s": {"type": "string"},
                    "n": {"type": "integer"},
                    "b": {"type": "boolean"},
                    "a": {"type": "array", "items": {"type": "string"}}
                  }
                }
              }
            }]
            """.utf8)
        let decoded = try! JSONSerialization.jsonObject(with: data) as! [[String: Any]]
        return decoded.map { $0.mapValues { asSendable($0) } }
    }()

    private func call(_ body: String, name: String = "f") -> String {
        "<function name=\"\(name)\">\(body)</function>"
    }

    @Test func schemaTypesAndUnknownStrings() {
        let xml = call("<param name=\"n\">3</param><param name=\"s\">007</param><param name=\"b\">false</param><param name=\"a\">[\"x\"]</param><param name=\"version\">3.10</param>")
        let parsed = parser.parse(content: xml, tools: Self.tools)
        #expect(parsed?.function.arguments["n"] == .int(3))
        #expect(parsed?.function.arguments["s"] == .string("007"))
        #expect(parsed?.function.arguments["b"] == .bool(false))
        #expect(parsed?.function.arguments["a"] == .array([.string("x")]))
        #expect(parsed?.function.arguments["version"] == .string("3.10"))
        #expect(parser.parse(content: xml, tools: nil)?.function.arguments["n"] == .string("3"))
        #expect(parser.parse(content: call("<param name=\"b\">banana</param>"), tools: Self.tools)?
            .function.arguments["b"] == .string("banana"))
        #expect(parser.parse(content: call("<param name=\"b\">True</param>"), tools: Self.tools)?
            .function.arguments["b"] == .bool(true))
        #expect(parser.parse(content: call("<param name=\"a\">['x', True, None]</param>"), tools: Self.tools)?
            .function.arguments["a"] == .array([.string("x"), .bool(true), .null]))
        let expression = "[__import__('os').system('not executed')]"
        #expect(parser.parse(content: call("<param name=\"a\">\(expression)</param>"), tools: Self.tools)?
            .function.arguments["a"] == .string(expression))
        #expect(parser.parse(content: call("<param name=\"a\">['x']</param>"), tools: nil)?
            .function.arguments["a"] == .string("['x']"))
    }

    @Test func cdataIsLiteralAndDoesNotEndTheEnvelope() {
        let value = "if a < b:\n  &x </function></param><think>literal</think>\n"
        let xml = call("<param name=\"s\"><![CDATA[\(value)]]></param>")
        #expect(parser.parse(content: xml, tools: Self.tools)?.function.arguments["s"] == .string(value))
        #expect(parser.completeToolCallEnd(in: xml) == xml.endIndex)
        #expect(parser.completeToolCallEnd(in: "<function name=\"f\"><param name=\"s\"><![CDATA[</function>") == nil)
    }

    @Test func reasoningExamplesAreNotCalls() {
        let ghost = call("", name: "ghost")
        let real = call("", name: "real")
        #expect(parser.parseEOS("<think>Consider \(ghost)</think>\(real)", tools: nil).map(\.function.name) == ["real"])
        #expect(parser.parseEOS("<think>Never closed \(ghost)", tools: nil).isEmpty)
        #expect(!ToolCallFormat.minicpm5.parsesToolCallsFromReasoningChannel)
    }

    @Test func malformedOrTruncatedCallsAreNotExecutable() {
        #expect(parser.parse(content: "<function name=\"f\"><param name=\"n\">3", tools: Self.tools) == nil)
        #expect(parser.parse(content: call("<param name=\"n\">1</param><param name=\"n\">2</param>"), tools: Self.tools) == nil)
        #expect(parser.parse(content: call("<param name=\"s\"><nested>bad</nested></param>"), tools: Self.tools) == nil)
        #expect(parser.parse(content: call("<param name=\"s\">&external;</param>"), tools: Self.tools) == nil)
    }

    @Test func persistedArgumentOrderAndValuesSurvive() throws {
        let parsed = try #require(parser.parse(
            content: call("<param name=\"s\">007</param><param name=\"n\">3</param>"), tools: Self.tools))
        let restored = try JSONDecoder().decode(ToolCall.self, from: JSONEncoder().encode(parsed))
        #expect(restored.function.argumentOrder == ["s", "n"])
        #expect(restored.function.arguments == parsed.function.arguments)
    }

    @Test func everyChunkSplitAndMultipleCalls() {
        let value = "</function> & <think>literal</think>"
        let xml = call("<param name=\"s\"><![CDATA[\(value)]]></param>") + call("", name: "g")
        let text = "Before " + xml + " After"
        // Every character boundary, including CDATA/open/close delimiters.
        for offset in 0...text.count {
            let split = text.index(text.startIndex, offsetBy: offset)
            let processor = ToolCallProcessor(format: .minicpm5, tools: Self.tools)
            var visible = processor.processChunk(String(text[..<split])) ?? ""
            visible += processor.processChunk(String(text[split...])) ?? ""
            visible += processor.processEOS() ?? ""
            #expect(processor.toolCalls.map(\.function.name) == ["f", "g"], "split \(offset)")
            #expect(processor.toolCalls.first?.function.arguments["s"] == .string(value), "split \(offset)")
            #expect(visible == "Before  After", "split \(offset): \(visible)")
            #expect(processor.toolCallProtocolFailure == nil)
        }
    }

    @Test func templateSignatureAndOtherDialectsStaySeparate() {
        let template = #"<function name="f"><param name="s"><![CDATA[x]]></param></function>"#
        #expect(MiniCPM5ToolCallParser.matchesTemplate(template))
        #expect(!MiniCPM5ToolCallParser.matchesTemplate("a <function name=\" mentioned in prose"))
        #expect(ToolCallFormat.fromCapabilityName("minicpm5") == .minicpm5)
        #expect(ToolCallFormat.fromCapabilityName("minicpm5_xml_function") == .minicpm5)
        #expect(ToolCallFormat.json.createParser().usesCustomEndBoundary == false)
        #expect(ToolCallFormat.xmlFunction.createParser().usesCustomEndBoundary == false)
    }

    @Test func reasoningThenToolPipelinePreservesLiteralPayload() {
        // Production uses the prompt-resolved parser, not the factory alone.
        var reasoning = ReasoningParser.forPrompt(stampName: "minicpm5", promptTail: "<|im_start|>assistant\n")!
        let processor = ToolCallProcessor(format: .minicpm5, tools: Self.tools)
        let literal = "save <think>literal</think> and </function> & data"
        let xml = call("<param name=\"s\"><![CDATA[\(literal)]]></param>")
        let stream = "<think>Need the file.</think>" + xml + "Saved."
        var thought = ""
        var visible = ""
        func consume(_ segments: [ReasoningSegment]) {
            for segment in segments {
                switch segment {
                case .reasoning(let text): thought += text
                case .content(let text): visible += processor.processChunk(text) ?? ""
                }
            }
        }
        for character in stream { consume(reasoning.feed(String(character))) }
        consume(reasoning.flush())
        visible += processor.processEOS() ?? ""
        #expect(thought == "Need the file.")
        #expect(visible == "Saved.")
        #expect(processor.toolCalls.first?.function.arguments["s"] == .string(literal))
    }

    @Test func nativeReasoningPromptStateAndUnfinishedEnvelopes() throws {
        for (tail, inside) in [("<|im_start|>assistant\n", false), ("<think>\n", true), ("<think>\n\n</think>\n\n", false)] {
            let parser = try #require(ReasoningParser.forPrompt(stampName: "minicpm5", promptTail: tail))
            #expect(parser.preservesXMLFunctionPayloads)
            #expect(parser.isInsideReasoning == inside)
        }
        #expect(ReasoningParser().preservesXMLFunctionPayloads == false)
        #expect(ReasoningParser.fromCapabilityName("qwen3")?.preservesXMLFunctionPayloads == false)
        var parser = try #require(ReasoningParser.forPrompt(stampName: "minicpm5", promptTail: "<|im_start|>assistant\n"))
        let text = "<function name=\"f\"><param name=\"s\"><![CDATA[<think>unfinished"
        let segments = parser.feed(text) + parser.flush()
        #expect(segments == [.content(text)])
        let processor = ToolCallProcessor(format: .minicpm5, tools: Self.tools)
        for case .content(let content) in segments { _ = processor.processChunk(content) }
        _ = processor.processEOS()
        #expect(processor.toolCalls.isEmpty)
    }
}
