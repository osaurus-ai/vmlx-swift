import Testing

import MLX
@testable import MLXLMCommon

private struct PromptTailTestTokenizer: Tokenizer {
    let pieces: [Int: String]

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { pieces[$0] ?? "" }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

@Suite("Prompt-tail decode for low-level generation")
struct PromptTailDecodeTests {
    @Test("token-id prompt tail without think tags starts in content")
    func tokenIDTailWithoutThinkTagsStartsInContent() throws {
        let tokenizer = PromptTailTestTokenizer(pieces: [
            1: "<role>HUMAN</role>Hello",
            2: "<role>ASSISTANT</role>",
        ])
        let tail = _decodePromptTail(
            tokenIds: [1, 2], tokenizer: tokenizer, tokens: 64)

        var optionalParser = ReasoningParser.forPrompt(
            stampName: "deepseek_r1", promptTail: tail)
        var parser = try #require(optionalParser)

        var content = ""
        var reasoning = ""
        for segment in parser.feed("Visible answer.") {
            switch segment {
            case .content(let text): content += text
            case .reasoning(let text): reasoning += text
            }
        }
        for segment in parser.flush() {
            switch segment {
            case .content(let text): content += text
            case .reasoning(let text): reasoning += text
            }
        }

        #expect(content == "Visible answer.")
        #expect(reasoning.isEmpty)
    }

    @Test("token-id prompt tail with open think starts in reasoning")
    func tokenIDTailWithOpenThinkStartsInReasoning() throws {
        let tokenizer = PromptTailTestTokenizer(pieces: [
            1: "<|im_start|>assistant\n",
            2: "<think>\n",
        ])
        let tail = _decodePromptTail(
            tokenIds: [1, 2], tokenizer: tokenizer, tokens: 64)

        var optionalParser = ReasoningParser.forPrompt(
            stampName: "qwen3_6", promptTail: tail)
        var parser = try #require(optionalParser)

        var reasoning = ""
        for segment in parser.feed("hidden thought") {
            if case .reasoning(let text) = segment { reasoning += text }
        }
        for segment in parser.flush() {
            if case .reasoning(let text) = segment { reasoning += text }
        }

        #expect(reasoning.contains("hidden thought"))
    }

    @Test("input token array tail with nil tokenIds keeps closed think content-visible")
    func inputTokenArrayTailWithNilTokenIdsKeepsClosedThinkContentVisible() throws {
        let tokenizer = PromptTailTestTokenizer(pieces: [
            1: "<|im_start|>assistant\n",
            2: "<think></think>",
        ])
        let input = LMInput(tokens: MLXArray([1, 2]))
        let tail = _decodePromptTail(input: input, tokenizer: tokenizer, tokens: 64)

        // `"mimo"` is a MODEL TYPE, not a capability stamp: `reasoningStampFromModelType` maps it
        // to `"think_xml"`, and that is what `fromCapabilityName` accepts — as
        // `MiMoV2FlashRuntimeTests` asserts directly. Passing the model type here resolved nil, so
        // this test force-unwrapped nil and took the whole process down with it. Go through the
        // mapping, which is what the runtime does.
        var optionalParser = ReasoningParser.forPrompt(
            stampName: reasoningStampFromModelType("mimo"), promptTail: tail)
        var parser = try #require(optionalParser)

        var content = ""
        var reasoning = ""
        for segment in parser.feed("Blue.") {
            switch segment {
            case .content(let text): content += text
            case .reasoning(let text): reasoning += text
            }
        }
        for segment in parser.flush() {
            switch segment {
            case .content(let text): content += text
            case .reasoning(let text): reasoning += text
            }
        }

        #expect(content == "Blue.")
        #expect(reasoning.isEmpty)
    }
}
