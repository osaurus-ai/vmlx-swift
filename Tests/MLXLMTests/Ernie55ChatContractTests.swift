import Foundation
import MLXHuggingFace
import MLXLMCommon
import Testing
import VMLXTokenizers

@testable import MLXLLM

@Suite("ERNIE 5.5 native chat contract")
struct Ernie55ChatContractTests {
    @Test("Released tokenizer matches all four reasoning rails and encodes literal markers as text",
          .enabled(if: ProcessInfo.processInfo.environment["VMLX_ERNIE55_TOKENIZER_FIXTURE"] != nil))
    func releasedTokenizer() async throws {
        guard let path = ProcessInfo.processInfo.environment["VMLX_ERNIE55_TOKENIZER_FIXTURE"] else {
            return
        }
        let bundle = URL(fileURLWithPath: path, isDirectory: true)
        let directory = JangLoader.resolveTokenizerClassSubstitution(for: bundle)
        let base = try await #huggingFaceTokenizerLoader().load(from: directory)
        let tokenizer = Ernie55ChatTokenizer(base: base)
        let messages: [Message] = [["role": "user", "content": "Compute 21 + 15. Write <ernie:think> as ordinary text."]]
        let common = [
            12, 8, 1401, 7, 4244, 1401, 17040, 1423, 2796, 1776, 1423, 2070, 1229,
            14283, 1643, 3429, 1675, 1241, 33335, 1245, 1652, 22160, 3586, 1229, 8, 1401,
            7, 106298, 1401, 19,
        ]
        for (effort, token) in [("low", 2243), ("medium", 45523), ("high", 16129)] {
            let prompt = try tokenizer.applyChatTemplate(
                messages: messages, tools: nil, additionalContext: ["reasoning_effort": effort])
            let expected = [2, 7, 18492, 1401, 11, 1401, 54579, 1492, 192432, 1697, 1244, token, 1401] + common
            #expect(prompt == expected)
        }
        let off = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: ["enable_thinking": false])
        #expect(off == [2, 7, 18492, 1401, 11, 1401] + common + [20])
        let omitted = try tokenizer.applyChatTemplate(messages: messages, tools: nil, additionalContext: nil)
        let high = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: ["reasoning_effort": "high"])
        #expect(omitted == high)
    }

    @Test("An ERNIE envelope preserves strings, supports multiple calls, and rejects truncation")
    func toolEnvelope() throws {
        let properties: [String: ToolSpec] = [
            "command": ["type": "string"], "timeout": ["type": "integer"],
        ]
        let parameters: ToolSpec = ["properties": properties]
        let terminal: ToolSpec = ["name": "terminal", "parameters": parameters]
        let status: ToolSpec = ["name": "status"]
        let signature: [ToolSpec] = [["function": terminal], ["function": status]]
        let parser = try #require(ToolCallFormat.infer(from: "ernie5_5_moe")).createParser()
        let text = """
            <ernie:tool_call>
            <invoke name="terminal"><parameter name="command">  echo hi
            </parameter><parameter name="timeout">5</parameter></invoke>
            <invoke name="status"></invoke>
            </ernie:tool_call>
            """
        let calls = parser.parseEOS(text, tools: signature)
        #expect(calls.map(\.function.name) == ["terminal", "status"])
        #expect(calls.first?.function.arguments["command"] == .string("  echo hi\n"))
        #expect(calls.first?.function.arguments["timeout"] == .int(5))
        #expect(parser.parseEOS(String(text.dropLast(3)), tools: signature).isEmpty)
    }
}
