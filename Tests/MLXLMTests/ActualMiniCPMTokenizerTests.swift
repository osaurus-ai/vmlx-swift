import Foundation
import Testing
import MLXLMCommon
import MLXHuggingFace
@preconcurrency import VMLXTokenizers

@Suite("Actual MiniCPM5 tokenizer bridge — no model weights", .serialized,
       .enabled(if: FileManager.default.fileExists(atPath: FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("models/OsaurusAI/MiniCPM5-2B-JANG_8M/tokenizer.json").path)))
struct ActualMiniCPMTokenizerTests {
    @Test func nativeToolsDoNotTriggerNemotronFallback() async throws {
        let directory = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("models/OsaurusAI/MiniCPM5-2B-JANG_8M")
        let tokenizer = try await #huggingFaceTokenizerLoader().load(
            from: JangLoader.resolveChatTemplateSidecarSubstitution(for: directory))
        let parameters: [String: any Sendable] = ["type": "object", "properties": ["path": ["type": "string"]], "required": ["path"]]
        let function: [String: any Sendable] = ["name": "read_file", "description": "Read a file", "parameters": parameters]
        let tools: [[String: any Sendable]] = [["type": "function", "function": function]]
        for thinking: Bool? in [nil, true, false] {
            let context: [String: any Sendable]? = thinking.map { ["enable_thinking": $0] }
            let ids = try tokenizer.applyChatTemplate(messages: [["role": "user", "content": "Read note.md"]], tools: tools, additionalContext: context)
            let rendered = tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
            #expect(rendered.contains("<function name="), "native XML grammar must survive the production bridge")
            #expect(!rendered.contains("<function="), "Nemotron fallback must not replace MiniCPM")
            let tail = "<|im_start|>assistant\n" + (thinking == true ? "<think>\n" : thinking == false ? "<think>\n\n</think>\n\n" : "")
            #expect(rendered.hasSuffix(tail), "native mode \(String(describing: thinking)) tail: \(rendered.suffix(100))")
            let controllable = try #require(tokenizer as? any GenerationPromptControllableTokenizer)
            let withoutTail = try controllable.applyChatTemplate(
                messages: [["role": "user", "content": "Read note.md"]], tools: tools,
                additionalContext: context, addGenerationPrompt: false)
            let historyOnly = tokenizer.decode(tokenIds: withoutTail, skipSpecialTokens: false)
            #expect(historyOnly.contains("<function name="))
            #expect(!historyOnly.contains("<function="))
            #expect(!historyOnly.hasSuffix(tail))
        }
    }
    @Test func persistedNativeToolArgumentsRenderInEmittedOrder() async throws {
        let directory = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("models/OsaurusAI/MiniCPM5-2B-JANG_8M")
        let tokenizer = try await #huggingFaceTokenizerLoader().load(
            from: JangLoader.resolveChatTemplateSidecarSubstitution(for: directory))
        let call = ToolCall(function: .init(name: "write", arguments: [
            "path": .string("note.md"), "content": .string("007")
        ], argumentOrder: ["path", "content"]))
        let restored = try JSONDecoder().decode(ToolCall.self, from: JSONEncoder().encode(call))
        let messages = [
            defaultMessageDict(for: .user("Write the note.")),
            defaultMessageDict(for: .assistant("", toolCalls: [restored]))
        ]
        let ids = try tokenizer.applyChatTemplate(messages: messages, tools: nil, additionalContext: nil)
        let rendered = tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
        #expect(rendered.contains(#"<function name="write"><param name="path">note.md</param><param name="content">007</param></function>"#))
        #expect(!rendered.contains("_vmlx_tool_argument_orders"))
        print("MINICPM_ORDER native tokenizer history tokens=\(ids.count) persisted order path,content")
    }

    @Test func nativePromptTailsAndEndToken() async throws {
        let directory = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("models/OsaurusAI/MiniCPM5-2B-JANG_8M")
        let template = try String(contentsOf: directory.appendingPathComponent("chat_template.jinja"), encoding: .utf8)
        #expect(!ChatTemplateRepair.needsRepair(template))
        #expect(ChatTemplateRepair.repaired(template) == template)
        let resolved = JangLoader.resolveChatTemplateSidecarSubstitution(for: directory)
        let tokenizer = try await #huggingFaceTokenizerLoader().load(from: resolved)
        #expect(tokenizer.convertTokenToId("<|im_end|>") == 130073)
        #expect(tokenizer.eosTokenId == 1)
        let messages: [[String: any Sendable]] = [["role": "user", "content": "What is 2 plus 3?"]]
        var tails: [String] = []
        for thinking: Bool? in [nil, true, false] {
            let context: [String: any Sendable]? = thinking.map { ["enable_thinking": $0] }
            let ids = try tokenizer.applyChatTemplate(messages: messages, tools: nil, additionalContext: context)
            let rendered = tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
            print("MINICPM_TOKENIZER thinking=\(String(describing: thinking)) count=\(ids.count) ids=\(ids) text=\(rendered.debugDescription)")
            tails.append(rendered)
        }
        #expect(tails[0].hasSuffix("<|im_start|>assistant\n"))
        #expect(tails[1] == tails[0] + "<think>\n")
        #expect(tails[2] == tails[0] + "<think>\n\n</think>\n\n")
        let defaults = try JSONDecoder().decode(GenerationConfigFile.self, from: Data(contentsOf: directory.appendingPathComponent("generation_config.json")))
        let sampling = GenerateParameters(generationConfig: defaults)
        #expect(sampling.temperature == 1)
        #expect(sampling.topP == 0.95)
        #expect(sampling.repetitionPenalty == nil)
        #expect(sampling.topK == 0)
    }
}
