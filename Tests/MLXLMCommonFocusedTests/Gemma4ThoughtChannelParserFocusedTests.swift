// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Testing
@testable import MLXLMCommon

@Suite("Gemma4 thought-channel parser focused contracts")
struct Gemma4ThoughtChannelParserFocusedTests {
    @Test("empty thought channel without newline does not surface thought")
    func emptyThoughtChannelWithoutNewlineDoesNotSurfaceThought() {
        var parser = ReasoningParser.fromCapabilityName("gemma4")
        let segments = feed("pre<|channel>thought<channel|>answer", into: &parser, chunkSize: 2)

        let (reasoning, content) = collect(segments)
        #expect(reasoning.isEmpty)
        #expect(content == "preanswer")
        for marker in ["thought", "<|channel>", "<channel|>"] {
            #expect(!reasoning.contains(marker))
            #expect(!content.contains(marker))
        }
    }

    @Test("pipe thought channel strips channel header from reasoning delta")
    func pipeThoughtChannelStripsChannelHeaderFromReasoningDelta() {
        var parser = ReasoningParser.fromCapabilityName("gemma4")
        let segments = feed("pre<|channel|>thought\nhidden plan<channel|>answer", into: &parser, chunkSize: 3)

        let (reasoning, content) = collect(segments)
        #expect(reasoning == "hidden plan")
        #expect(content == "preanswer")
        for marker in ["thought", "<|channel|>", "<channel|>"] {
            #expect(!reasoning.contains(marker))
            #expect(!content.contains(marker))
        }
    }

    @Test("prompt-tail open thought channel strips repeated thought header")
    func promptTailOpenThoughtChannelStripsRepeatedThoughtHeader() {
        var parser = ReasoningParser.forPrompt(
            stampName: "gemma4",
            promptTail: "<start_of_turn>model\n<|channel>thought\n")
        let segments = feed("thought\nK7Q: yellow\nR9Z: purple<channel|>", into: &parser, chunkSize: 4)

        let (reasoning, content) = collect(segments)
        #expect(reasoning == "K7Q: yellow\nR9Z: purple")
        #expect(content.isEmpty)
        #expect(!reasoning.hasPrefix("thought"))
    }

    @Test("prompt-tail open thought channel drops repeated empty thought header before visible content")
    func promptTailOpenThoughtChannelDropsRepeatedEmptyThoughtHeaderBeforeVisibleContent() {
        var parser = ReasoningParser.forPrompt(
            stampName: "gemma4",
            promptTail: "<start_of_turn>model\n<|channel>thought\n")
        let segments = feed("thought\n<channel|>NO_IMAGE_VISIBLE", into: &parser, chunkSize: 4)

        let (reasoning, content) = collect(segments)
        #expect(reasoning.isEmpty)
        #expect(content == "NO_IMAGE_VISIBLE")
    }

    @Test("prompt-tail open thought channel drops repeated empty thought header at stream end")
    func promptTailOpenThoughtChannelDropsRepeatedEmptyThoughtHeaderAtStreamEnd() {
        var parser = ReasoningParser.forPrompt(
            stampName: "gemma4",
            promptTail: "<start_of_turn>model\n<|channel>thought\n")
        let segments = feed("thought\n", into: &parser, chunkSize: 4)

        let (reasoning, content) = collect(segments)
        #expect(reasoning.isEmpty)
        #expect(content.isEmpty)
    }

    private func feed(
        _ text: String,
        into parser: inout ReasoningParser?,
        chunkSize: Int
    ) -> [ReasoningSegment] {
        var segments: [ReasoningSegment] = []
        var index = text.startIndex
        while index < text.endIndex {
            let next = text.index(index, offsetBy: chunkSize, limitedBy: text.endIndex) ?? text.endIndex
            segments.append(contentsOf: parser?.feed(String(text[index..<next])) ?? [])
            index = next
        }
        segments.append(contentsOf: parser?.flush() ?? [])
        return segments
    }

    private func collect(_ segments: [ReasoningSegment]) -> (reasoning: String, content: String) {
        var reasoning = ""
        var content = ""
        for segment in segments {
            switch segment {
            case .reasoning(let text):
                reasoning += text
            case .content(let text):
                content += text
            }
        }
        return (reasoning, content)
    }
}
