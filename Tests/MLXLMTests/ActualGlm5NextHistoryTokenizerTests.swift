import Foundation
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Testing
@preconcurrency import VMLXTokenizers

/// Opt into local native sidecars explicitly. No model weights or downloads.
/// The strict proof invocation also supplies the retained three-turn receipt.
@Suite("Actual GLM history tokenizer and cache boundaries", .serialized,
    .enabled(if: ProcessInfo.processInfo.environment["VMLX_GLM5_NATIVE_TOKENIZER_DIR"] != nil))
struct ActualGlm5NextHistoryTokenizerTests {
    private func load() async throws -> (any MLXLMCommon.Tokenizer, Glm5NextProcessor) {
        let path = try #require(ProcessInfo.processInfo.environment["VMLX_GLM5_NATIVE_TOKENIZER_DIR"])
        let directory = URL(fileURLWithPath: path, isDirectory: true)
        for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json"] {
            #expect(FileManager.default.fileExists(atPath: directory.appendingPathComponent(name).path))
        }
        let template = try String(contentsOf: directory.appendingPathComponent("chat_template.jinja"), encoding: .utf8)
        #expect(!ChatTemplateRepair.needsRepair(template))
        #expect(ChatTemplateRepair.repaired(template) == template)
        let tokenizer = try await #huggingFaceTokenizerLoader().load(
            from: JangLoader.resolveChatTemplateSidecarSubstitution(for: directory))
        let config = try JSONDecoder().decode(Glm5NextProcessorConfiguration.self,
            from: Data(contentsOf: directory.appendingPathComponent("processor_config.json")))
        return (tokenizer, try Glm5NextProcessor(config: config, tokenizer: tokenizer))
    }

    private static var tools: [[String: any Sendable]] {
        let readProperties: [String: any Sendable] = [
            "path": ["type": "string"], "max_lines": ["type": "integer"],
        ]
        let writeProperties: [String: any Sendable] = [
            "path": ["type": "string"], "content": ["type": "string"],
        ]
        let readParameters: [String: any Sendable] = ["type": "object", "properties": readProperties]
        let writeParameters: [String: any Sendable] = ["type": "object", "properties": writeProperties]
        let read: [String: any Sendable] = [
            "name": "read_file", "description": "Read a file", "parameters": readParameters,
        ]
        let write: [String: any Sendable] = [
            "name": "write_file", "description": "Write a file", "parameters": writeParameters,
        ]
        return [["type": "function", "function": read], ["type": "function", "function": write]]
    }

    private func render(_ tokenizer: any MLXLMCommon.Tokenizer, _ history: [Chat.Message],
        tools: [[String: any Sendable]]? = nil,
        context: [String: any Sendable]? = nil) throws -> String {
        let ids = try tokenizer.applyChatTemplate(
            messages: Glm5NextMessageGenerator().generate(messages: history),
            tools: tools, additionalContext: context)
        return tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
    }

    @Test func nativeReasoningPolicyDecidesWhatHistoryRetains() async throws {
        let (tokenizer, _) = try await load()
        let history: [Chat.Message] = [
            .user("What was inspected?"),
            .init(role: .assistant, content: "note.md", reasoningContent: "Inspect note.md before editing."),
            .user("Continue."),
        ]
        let native = try render(tokenizer, history)
        #expect(native.contains("<think>Inspect note.md before editing.</think>note.md"))
        let cleared = try render(tokenizer, history, context: ["clear_thinking": true])
        #expect(!cleared.contains("Inspect note.md before editing."))
        #expect(cleared.contains("<think></think>note.md"))
        #expect(native.hasSuffix("<|assistant|><think>"))
        #expect(cleared.hasSuffix("<|assistant|><think>"))
    }

    @Test func nativeDependentCallsKeepArgumentsResultsAndReasoning() async throws {
        let (tokenizer, _) = try await load()
        let history = Glm5NextMessageHistoryTests.history()
        let native = try render(tokenizer, history, tools: Self.tools)
        #expect(native.contains("<think>Inspect the existing file before changing it.</think>"))
        #expect(native.contains("<think>The read returned 007; use that exact string.</think>"))
        let read = "<tool_call>read_file<arg_key>path</arg_key><arg_value>note.md</arg_value><arg_key>max_lines</arg_key><arg_value>7</arg_value></tool_call>"
        let write = "<tool_call>write_file<arg_key>path</arg_key><arg_value>note.md</arg_value><arg_key>content</arg_key><arg_value>007</arg_value></tool_call>"
        let readRange = try #require(native.range(of: read))
        let resultRange = try #require(native.range(of: "<tool_response>The inspected identifier is 007.</tool_response>"))
        let writeRange = try #require(native.range(of: write))
        #expect(readRange.upperBound <= resultRange.lowerBound)
        #expect(resultRange.upperBound <= writeRange.lowerBound)
        #expect(native.contains("<tool_response>Written 007 to note.md.</tool_response>"))
        #expect(!native.contains("_vmlx_tool_argument_orders"))
        let persisted = try JSONDecoder().decode(ToolCall.self,
            from: JSONEncoder().encode(Glm5NextMessageHistoryTests.writeCall()))
        var restoredHistory = history
        restoredHistory[3].toolCalls = [persisted]
        #expect(try render(tokenizer, restoredHistory, tools: Self.tools) == native)
    }

    @Test func nativeToolResultIDsRestoreCallOrder() async throws {
        let (tokenizer, _) = try await load()
        let history: [Chat.Message] = [
            .user("Read and write the note."),
            .assistant("", toolCalls: [Glm5NextMessageHistoryTests.readCall(), Glm5NextMessageHistoryTests.writeCall()]),
            .tool("WRITE_RESULT", toolCallId: "call_write"),
            .tool("READ_RESULT", toolCallId: "call_read"),
        ]
        let native = try render(tokenizer, history, tools: Self.tools)
        let read = try #require(native.range(of: "<tool_response>READ_RESULT</tool_response>"))
        let write = try #require(native.range(of: "<tool_response>WRITE_RESULT</tool_response>"))
        #expect(read.upperBound <= write.lowerBound)
    }

    private struct Receipt: Decodable, Sendable {
        struct Turn: Decodable, Sendable { let user: String; let visible: String; let reasoning: String }
        let turns: [Turn]
    }

    @Test func retainedNativeProsePromptsPublishExactGrowingPrefixes() async throws {
        let (tokenizer, processor) = try await load()
        let receiptPath = try #require(ProcessInfo.processInfo.environment["VMLX_GLM5_NATIVE_CONVERSATION_RECEIPT"])
        let receipt = try JSONDecoder().decode(Receipt.self, from: Data(contentsOf: URL(fileURLWithPath: receiptPath)))
        #expect(receipt.turns.count == 3)
        try await MLXMetalTestLock.withLock {
            var history: [Chat.Message] = []
            var previousPrompt: [Int] = []
            var previousBoundary: [Int] = []
            for (index, turn) in receipt.turns.enumerated() {
                history.append(.user(turn.user))
                let messages = Glm5NextMessageGenerator().generate(messages: history)
                let prompt = try tokenizer.applyChatTemplate(messages: messages, tools: nil, additionalContext: nil)
                let boundaries = canonicalChatCacheBoundaries(tokenizer: tokenizer, messages: messages,
                    tools: nil, additionalContext: nil, promptTokens: prompt)
                let boundary = try #require(boundaries.all.max())
                #expect(boundary > 0 && boundary < prompt.count)
                let prepared = try await processor.prepare(input: UserInput(chat: history))
                #expect(prepared.text.tokenIds == prompt)
                #expect(prepared.text.tokens.asArray(Int32.self).map(Int.init) == prompt)
                #expect(prepared.text.tokens.dtype == .int32)
                #expect(prepared.cachePrefixTokenCounts == boundaries.all)
                #expect(prepared.cacheStablePrefixTokenCounts == boundaries.stable)
                if index > 0 {
                    #expect(prompt.prefix(previousPrompt.count).elementsEqual(previousPrompt))
                    #expect(prompt.prefix(previousBoundary.count).elementsEqual(previousBoundary))
                    #expect(boundary > previousBoundary.count)
                }
                print("GLM_NATIVE_PREFIX run=\(index) prompt=\(prompt.count) all=\(boundaries.all) stable=\(boundaries.stable)")
                previousPrompt = prompt
                previousBoundary = Array(prompt.prefix(boundary))
                history.append(.init(role: .assistant, content: turn.visible, reasoningContent: turn.reasoning))
            }
        }
    }

    @Test func dependentToolPromptsPublishExactContinuationPrefixes() async throws {
        let (tokenizer, processor) = try await load()
        try await MLXMetalTestLock.withLock {
            let history = Glm5NextMessageHistoryTests.history()
            var previousBoundary: [Int] = []
            for count in [1, 3, 5] {
                let prefix = Array(history.prefix(count))
                let input = UserInput(chat: prefix, tools: Self.tools)
                let messages = Glm5NextMessageGenerator().generate(from: input)
                let prompt = try tokenizer.applyChatTemplate(messages: messages, tools: input.tools, additionalContext: nil)
                let boundaries = canonicalChatCacheBoundaries(tokenizer: tokenizer, messages: messages,
                    tools: input.tools, additionalContext: nil, promptTokens: prompt)
                let top = try #require(boundaries.all.max())
                #expect(top > 0 && top < prompt.count)
                #expect(prompt.prefix(previousBoundary.count).elementsEqual(previousBoundary))
                let prepared = try await processor.prepare(input: input)
                #expect(prepared.text.tokenIds == prompt)
                #expect(prepared.cachePrefixTokenCounts == boundaries.all)
                #expect(prepared.toolSchemas?.count == Self.tools.count)
                previousBoundary = Array(prompt.prefix(top))
                print("GLM_NATIVE_TOOL_PREFIX history=\(count) prompt=\(prompt.count) all=\(boundaries.all)")
            }
        }
    }

    @Test func nativeMediaMarkersSurviveHistoryMapping() async throws {
        let (tokenizer, _) = try await load()
        let markerHistory: [Chat.Message] = [.user("Inspect.",
            images: [.url(URL(fileURLWithPath: "/fixture/image.png"))],
            videos: [.url(URL(fileURLWithPath: "/fixture/video.mp4"))])]
        let native = try render(tokenizer, markerHistory)
        #expect(native.contains("<|begin_of_image|><|image|><|end_of_image|>"))
        #expect(native.contains("<|begin_of_video|><|video|><|end_of_video|>"))
    }
}
