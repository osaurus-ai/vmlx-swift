import Foundation
import Testing
import MLXLMCommon
import MLXHuggingFace
@preconcurrency import VMLXTokenizers

@Suite("Spark2.5 declared tool dialect", .serialized)
struct Spark25DialectTests {
    @Test func eosCannotPublishAnUnfinishedValueOrEnvelope() {
        for raw in [
            "<tool_call>write_note<arg_key>text</arg_key><arg_value>literal </tool_call>",
            "<tool_call>write_note<arg_key>text</arg_key><arg_value>note</arg_value>",
        ] {
            let processor = ToolCallProcessor(format: .glm4)
            for char in raw { _ = processor.processChunk(String(char)) }
            #expect(processor.toolCalls.isEmpty)
            _ = processor.processEOS()
            #expect(processor.toolCalls.isEmpty)
            #expect(processor.toolCallProtocolFailure == .malformedEnvelope)
        }
    }

    @Test func literalClosersAndAdjacentCallsKeepTheirBoundaries() throws {
        let value = "Literal <tool_call>example</tool_call> and <arg_key>key</arg_key>."
        let first = "<tool_call>write_note<arg_key>text</arg_key><arg_value>\(value)</arg_value></tool_call>"
        let raw = first + "<tool_call>lookup_marker</tool_call>"
        let parser = GLM4ToolCallParser()
        #expect(parser.parseEOS(raw, tools: nil).count == 2)
        for split in 0...raw.count {
            let index = raw.index(raw.startIndex, offsetBy: split)
            let processor = ToolCallProcessor(format: .glm4)
            _ = processor.processChunk(String(raw[..<index]))
            _ = processor.processChunk(String(raw[index...]))
            #expect(processor.toolCalls.map(\.function.name) == ["write_note", "lookup_marker"], "split \(split)")
            #expect(processor.toolCalls.first?.function.arguments["text"] == .string(value))
            #expect(processor.toolCallProtocolFailure == nil)
        }
    }

    @Test func malformedStructureOutsideValuesIsNotReinterpreted() {
        let pair = "<arg_key>text</arg_key><arg_value>note</arg_value>"
        for body in [
            "write_note<tool_call>other\(pair)",
            "write_note\(pair)<tool_call>other",
            "write_note<arg_key>text</arg_key><tool_call>other<arg_value>note</arg_value>",
            "write_note\(pair)unparsed suffix",
            "write_note\(pair)\(pair)",
        ] {
            let raw = "<tool_call>\(body)</tool_call>"
            #expect(GLM4ToolCallParser().parse(content: raw, tools: nil) == nil)
            #expect(GLM4ToolCallParser().parseEOS(raw, tools: nil).isEmpty)
            let processor = ToolCallProcessor(format: .glm4)
            for char in raw { _ = processor.processChunk(String(char)) }
            #expect(processor.toolCalls.isEmpty)
            #expect(processor.toolCallProtocolFailure == .malformedEnvelope)
        }
    }

    @Test func literalToolStartMarkerInsideStringArgumentIsNotDeleted() throws {
        let value = "Document <tool_call> as the opening marker."
        let body = "write_note<arg_key>text</arg_key><arg_value>\(value)</arg_value>"
        let function: [String: any Sendable] = [
            "name": "write_note", "parameters": ["type": "object",
                "properties": ["text": ["type": "string"]], "required": ["text"]] as [String: any Sendable]]
        let tools: [[String: any Sendable]] = [["type": "function", "function": function]]
        for input in [body, "<tool_call>\(body)</tool_call>"] {
            let call = try #require(GLM4ToolCallParser().parse(content: input, tools: tools))
            #expect(call.function.arguments["text"] == .string(value))
        }
        let raw = "<tool_call>\(body)</tool_call>"
        for split in 0...raw.count {
            let index = raw.index(raw.startIndex, offsetBy: split)
            let processor = ToolCallProcessor(format: .glm4, tools: tools)
            _ = processor.processChunk(String(raw[..<index]))
            _ = processor.processChunk(String(raw[index...]))
            #expect(processor.toolCalls.count == 1)
            #expect(processor.toolCalls.first?.function.arguments["text"] == .string(value), "split \(split)")
            #expect(processor.toolCallProtocolFailure == nil)
        }
    }

    @Test func incompleteArgumentCannotPublishAPartialCall() {
        let processor = ToolCallProcessor(format: .glm4)
        _ = processor.processChunk("<tool_call>read_file<arg_key>path</arg_key><arg_value>note.md</arg_value><arg_key>limit</arg_key></tool_call>")
        #expect(processor.toolCalls.isEmpty)
        #expect(processor.toolCallProtocolFailure == .malformedEnvelope)
    }

    @Test func nativeCallSurvivesEveryTwoChunkBoundary() throws {
        let text = "<tool_call>read_file<arg_key>path</arg_key><arg_value>note.md</arg_value></tool_call>"
        for offset in 0...text.count {
            let index = text.index(text.startIndex, offsetBy: offset)
            let processor = ToolCallProcessor(format: .glm4)
            _ = processor.processChunk(String(text[..<index]))
            _ = processor.processChunk(String(text[index...]))
            #expect(processor.toolCalls.count == 1, "split \(offset)")
            #expect(processor.toolCalls.first?.function.arguments["path"] == .string("note.md"))
            #expect(processor.toolCallProtocolFailure == nil)
        }
    }

    @Test func promptOwnedReasoningOpenerNeedsNoSyntheticOutputTag() throws {
        for mode in [true, false] {
            var parser = try #require(ReasoningParser.forPrompt(
                stampName: "qwen3", promptTail: "<|Bot|>" + (mode ? "<think>" : "</think>")))
            let output = mode ? "Check the result.</think>Done." : "Done."
            var reasoning = ""
            var content = ""
            var segments: [ReasoningSegment] = []
            for char in output { segments += parser.feed(String(char)) }
            segments += parser.flush()
            for segment in segments {
                switch segment {
                case .reasoning(let text): reasoning += text
                case .content(let text): content += text
                }
            }
            #expect(content == "Done.")
            #expect(reasoning == (mode ? "Check the result." : ""))
        }
    }

    @Test func bundleAndArchitectureResolveTheNativeArgumentDialect() throws {
        #expect(ToolCallFormat.infer(from: "spark2_5") == .glm4)
        let format = try #require(ToolCallFormat.fromCapabilityName("spark25"))
        #expect(format == .glm4)
        let properties: [String: any Sendable] = [
            "path": ["type": "string"], "options": ["type": "object"]]
        let parameters: [String: any Sendable] = [
            "type": "object", "properties": properties, "required": ["path"]]
        let function: [String: any Sendable] = ["name": "read_file", "parameters": parameters]
        let tools: [[String: any Sendable]] = [["type": "function", "function": function]]
        let call = try #require(format.createParser().parse(
            content: #"<tool_call>read_file<arg_key>path</arg_key><arg_value>007</arg_value><arg_key>options</arg_key><arg_value>{"enabled":true,"items":[1,2]}</arg_value></tool_call>"#,
            tools: tools))
        #expect(call.function.name == "read_file")
        #expect(call.function.arguments["path"] == .string("007"))
        #expect(call.function.arguments["options"] == .object([
            "enabled": .bool(true), "items": .array([.int(1), .int(2)])
        ]))
    }
}

/// Optional local bundle proof: tokenizer/config only, never model weights.
@Suite("Actual Spark2.5 native tokenizer bridge", .serialized,
       .enabled(if: FileManager.default.fileExists(atPath: FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("models/JANGQ-AI/Spark-X2.5-4B-JANG_8M/tokenizer.json").path)))
struct Spark25NativeTokenizerTests {
    private var directory: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("models/JANGQ-AI/Spark-X2.5-4B-JANG_8M")
    }

    @Test func nativeModeTailsAndSpecialTokens() async throws {
        let tokenizer = try await #huggingFaceTokenizerLoader().load(
            from: JangLoader.resolveChatTemplateSidecarSubstitution(for: directory))
        #expect(tokenizer.eosTokenId == 1)
        for (token, id) in [("<think>", 3), ("</think>", 4), ("<tool_call>", 130977),
                            ("</tool_call>", 130984), ("<arg_key>", 130980)] {
            #expect(tokenizer.convertTokenToId(token) == id)
        }
        let expected = "<｜start▁of▁sentence｜><|System|>\nyou are a helpful assistant."
            + "<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|User|>What is 2 plus 3?"
            + "<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|>"
        for thinking: Bool? in [nil, true, false] {
            let ids = try tokenizer.applyChatTemplate(
                messages: [["role": "user", "content": "What is 2 plus 3?"]], tools: nil,
                additionalContext: thinking.map { ["enable_thinking": $0] })
            let rendered = tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
            #expect(rendered == expected + (thinking == false ? "</think>" : "<think>"))
            print("SPARK_NATIVE mode=\(String(describing: thinking)) ids=\(ids) text=\(rendered.debugDescription)")
        }
    }

    @Test func nativeToolsKeepTheCompleteParameterContract() async throws {
        let tokenizer = try await #huggingFaceTokenizerLoader().load(
            from: JangLoader.resolveChatTemplateSidecarSubstitution(for: directory))
        let options: [String: any Sendable] = [
            "type": "object", "properties": ["enabled": ["type": "boolean"]]]
        let properties: [String: any Sendable] = ["path": ["type": "string"], "options": options]
        let parameters: [String: any Sendable] = [
            "type": "object", "properties": properties, "required": ["path"]]
        let function: [String: any Sendable] = [
            "name": "read_file", "description": "Read a file", "parameters": parameters]
        let tools: [[String: any Sendable]] = [["type": "function", "function": function]]
        for thinking: Bool? in [nil, true, false] {
            let ids = try tokenizer.applyChatTemplate(messages: [["role": "user", "content": "Read note.md"]],
                tools: tools, additionalContext: thinking.map { ["enable_thinking": $0] })
            let text = tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
            let start = try #require(text.range(of: "<tools>\n"))
            let end = try #require(text.range(of: "\n</tools>"))
            let actual = try JSONSerialization.jsonObject(with: Data(text[start.upperBound..<end.lowerBound].utf8)) as? NSDictionary
            let expected = try JSONSerialization.jsonObject(with: JSONSerialization.data(withJSONObject: function)) as? NSDictionary
            #expect(actual == expected)
            #expect(text.hasPrefix("<｜start▁of▁sentence｜><|System|>\nyou are a helpful assistant.## Tools\n"))
            #expect(text.hasSuffix("<|Bot|>" + (thinking == false ? "</think>" : "<think>")))
            #expect(!text.contains("<function="))
            print("SPARK_CATALOG mode=\(String(describing: thinking)) tokens=\(ids.count) text=\(text.debugDescription)")
        }
    }

    @Test func nativeToolHistoryPreservesCallAndConsecutiveResults() async throws {
        let tokenizer = try await #huggingFaceTokenizerLoader().load(
            from: JangLoader.resolveChatTemplateSidecarSubstitution(for: directory))
        let call = ToolCall(function: .init(name: "read_file", arguments: ["path": .string("note.md")]))
        let messages = [defaultMessageDict(for: .user("Read the note.")),
            defaultMessageDict(for: .assistant("", toolCalls: [call])),
            ["role": "tool", "content": "first"], ["role": "tool", "content": "second"]]
        let ids = try tokenizer.applyChatTemplate(messages: messages, tools: nil, additionalContext: nil)
        let text = tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
        #expect(text.contains("<|Bot|></think><tool_call>read_file<arg_key>path</arg_key><arg_value>note.md</arg_value></tool_call><｜end▁of▁sentence｜>"))
        #expect(text.contains("<|Tool|><tool_response>first</tool_response><tool_response>second</tool_response><｜end▁of▁sentence｜>"))
        #expect(text.hasSuffix("<|Bot|><think>"))
        print("SPARK_HISTORY tokens=\(ids.count) text=\(text.debugDescription)")
    }
}
