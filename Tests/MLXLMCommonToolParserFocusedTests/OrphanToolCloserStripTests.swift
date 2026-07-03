// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Testing
@testable import MLXLMCommon

/// Orphan tool-call CLOSER tags must never leak as visible text.
///
/// Live Zaya/AppleScript rows (MODEL_ISSUES_TRIAGE Issue 3) emit orphan
/// closing tags — `</parameter></function></zyphra_tool_call>` with no
/// matching opener — after several agent-loop steps. The streaming
/// `ToolCallProcessor` state machine only entered tool-call collection after
/// matching an OPEN tag, so an orphan closer fell through the
/// `.potentialToolCall` flush and streamed to the user as literal protocol
/// text. These are protocol control markers for the format (for ZAYA the
/// wrapper tags are dedicated special tokens 101/102), the same robustness
/// class as the Gemma `<channel|>` stray-tag strip in `ReasoningParser`.
///
/// The strip is scoped to the format's OWN registered closers
/// (`orphanStripTags`) — arbitrary tag-looking prose (`</div>`) must still
/// pass through untouched, and real envelopes must keep parsing.
@Suite("Orphan tool-call closer strip")
struct OrphanToolCloserStripFocusedTests {

    // MARK: - The Issue 3 leak shape (zayaXml)

    @Test("zayaXml: an orphan closer run streams invisible, char-by-char")
    func zayaOrphanCloserRunStrippedCharByChar() {
        let output = "Volume set to 30.\n</parameter>\n</function>\n</zyphra_tool_call>"
        let processor = ToolCallProcessor(format: .zayaXml, tools: [lineCountToolSpec()])
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines) == "Volume set to 30.")
        #expect(!visible.contains("</parameter>"))
        #expect(!visible.contains("</function>"))
        #expect(!visible.contains("</zyphra_tool_call>"))
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("zayaXml: back-to-back orphan closers in one chunk strip, prose around them survives")
    func zayaOrphanClosersSingleChunk() {
        let processor = ToolCallProcessor(format: .zayaXml, tools: [lineCountToolSpec()])
        var visible = ""
        visible += processor.processChunk("Done.</parameter></function></zyphra_tool_call> All set.") ?? ""
        visible += processor.processEOS() ?? ""

        #expect(visible == "Done. All set.")
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("zayaXml: an orphan closer split across chunk boundaries still strips")
    func zayaOrphanCloserSplitAcrossChunks() {
        let processor = ToolCallProcessor(format: .zayaXml, tools: [lineCountToolSpec()])
        var visible = ""
        for chunk in ["ok </zy", "phra_tool", "_call>", " done"] {
            visible += processor.processChunk(chunk) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(visible == "ok  done")
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("zayaXml: a partial orphan closer at EOS is suppressed, not leaked")
    func zayaPartialOrphanCloserAtEOS() {
        let processor = ToolCallProcessor(format: .zayaXml, tools: [lineCountToolSpec()])
        var visible = ""
        visible += processor.processChunk("answer 42 ") ?? ""
        visible += processor.processChunk("</functi") ?? ""
        visible += processor.processEOS() ?? ""

        #expect(visible == "answer 42 ")
        #expect(processor.toolCalls.isEmpty)
    }

    // MARK: - No over-stripping

    @Test("zayaXml: tag-looking prose that is NOT a registered closer passes through")
    func zayaUnregisteredTagsPassThrough() {
        let processor = ToolCallProcessor(format: .zayaXml, tools: [lineCountToolSpec()])
        var visible = ""
        for ch in "HTML uses </div> and </p> to close elements." {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(visible == "HTML uses </div> and </p> to close elements.")
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("zayaXml: a real complete envelope still parses to a tool call with no leak")
    func zayaRealEnvelopeStillParses() {
        let output = """
        <zyphra_tool_call>
        <function=line_count>
        <parameter=text>
        red
        green
        </parameter>
        </function>
        </zyphra_tool_call>
        """
        let processor = ToolCallProcessor(format: .zayaXml, tools: [lineCountToolSpec()])
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "line_count")
        #expect(processor.toolCalls.first?.function.arguments["text"] == .string("red\ngreen"))
    }

    @Test("zayaXml: orphan closers AFTER a real envelope strip while the call still parses")
    func zayaOrphanClosersAfterRealEnvelope() {
        let output = "<zyphra_tool_call>\n<function=line_count>\n<parameter=text>\nhi\n</parameter>\n</function>\n</zyphra_tool_call>\n</function>\n</zyphra_tool_call>"
        let processor = ToolCallProcessor(format: .zayaXml, tools: [lineCountToolSpec()])
        var visible = ""
        for ch in output {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "line_count")
    }

    // MARK: - gemma4 (the shipped AppleScript 16B transport, zyphra-aliased)

    @Test("gemma4: orphan zyphra closers strip on the Gemma4 parser path")
    func gemma4OrphanZyphraClosersStrip() {
        let processor = ToolCallProcessor(format: .gemma4, tools: [lineCountToolSpec()])
        var visible = ""
        for ch in "Saved the note.</parameter></function></zyphra_tool_call>" {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(visible == "Saved the note.")
        #expect(processor.toolCalls.isEmpty)
    }

    @Test("gemma4: an orphan native closer strips too")
    func gemma4OrphanNativeCloserStrips() {
        let processor = ToolCallProcessor(format: .gemma4, tools: [lineCountToolSpec()])
        var visible = ""
        for ch in "Done.<tool_call|> next" {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(visible == "Done. next")
        #expect(processor.toolCalls.isEmpty)
    }

    // MARK: - Strip-only mode (no tools offered)

    @Test("zayaXml strip-only: orphan closers still strip and no calls are recorded")
    func zayaStripOnlyOrphanClosers() {
        let processor = ToolCallProcessor(format: .zayaXml, tools: nil, stripOnly: true)
        var visible = ""
        for ch in "hello </zyphra_tool_call> world" {
            visible += processor.processChunk(String(ch)) ?? ""
        }
        visible += processor.processEOS() ?? ""

        #expect(visible == "hello  world")
        #expect(processor.toolCalls.isEmpty)
    }

    private func lineCountToolSpec() -> [String: any Sendable] {
        [
            "type": "function",
            "function": [
                "name": "line_count",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "text": ["type": "string"] as [String: any Sendable],
                    ] as [String: any Sendable],
                    "required": ["text"],
                    "additionalProperties": false,
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }
}
