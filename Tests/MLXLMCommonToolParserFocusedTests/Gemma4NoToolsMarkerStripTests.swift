import Foundation
import Testing
@testable import MLXLMCommon

@Suite("Gemma4 tool-marker stripping with no tools in scope")
struct Gemma4NoToolsMarkerStripTests {
    // Reproduces the leak: thinking-on + zero tools offered + model
    // hallucinates a Gemma tool call -> control markers must NOT reach
    // visible text. Feeds the exact streamed body observed live.
    @Test("Gemma4 control markers are stripped from visible text when no tools are offered")
    func stripsMarkersWithoutTools() {
        let processor = ToolCallProcessor(format: .gemma4, tools: nil, stripOnly: true)
        var visible = ""
        for chunk in ["<|tool_call>", "call:osaurus_status{}", "<tool_call|>"] {
            if let out = processor.processChunk(chunk) { visible += out }
        }
        if let tail = processor.processEOS() { visible += tail }
        #expect(!visible.contains("<|tool_call>"), "leaked start marker: \(visible)")
        #expect(!visible.contains("<tool_call|>"), "leaked end marker: \(visible)")
        #expect(!visible.contains("call:"), "leaked bare-call marker: \(visible)")
        // No tools were offered, so no tool call should be surfaced.
        #expect(processor.toolCalls.isEmpty, "must not fabricate tool calls when none offered")
    }

    // The agent/sandbox loop sends an empty `tools` field (the schema rides in
    // the system prompt) yet the model emits a real `capabilities_load` call in
    // the drifted paren-JSON envelope. With no tools the strip-only processor
    // must still consume the whole envelope rather than leak the raw markup —
    // the exact symptom seen live in the chat UI.
    @Test("Gemma4 strips the paren-JSON capabilities_load envelope when no tools are offered")
    func stripsParenJSONEnvelopeWithoutTools() {
        let processor = ToolCallProcessor(format: .gemma4, tools: nil, stripOnly: true)
        let envelope =
            #"<|tool_call>call:capabilities_load({"ids": ["skill/Osaurus Browser", "tool/browser_navigate"]})<tool_call|>"#
        var visible = ""
        // Feed in small chunks to exercise the streaming state machine.
        var idx = envelope.startIndex
        while idx < envelope.endIndex {
            let end = envelope.index(idx, offsetBy: 3, limitedBy: envelope.endIndex) ?? envelope.endIndex
            if let out = processor.processChunk(String(envelope[idx..<end])) { visible += out }
            idx = end
        }
        if let tail = processor.processEOS() { visible += tail }
        #expect(!visible.contains("tool_call"), "leaked tool-call markup: \(visible)")
        #expect(!visible.contains("call:"), "leaked bare-call marker: \(visible)")
        #expect(processor.toolCalls.isEmpty, "must not fabricate tool calls when none offered")
    }
}
