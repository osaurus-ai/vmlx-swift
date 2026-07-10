// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Testing

@testable import MLXLMCommon

/// Prose that precedes a tool-call envelope must survive when the envelope
/// completes at end-of-sequence.
///
/// Qwen3.5-family models (Ornith, Qwen3.6) stop generating right after
/// `</tool_call>`, and the streaming pipeline's trailing holdbacks mean the
/// envelope's last characters routinely arrive only in the end-of-stream
/// flush — so the call completes inside `processEOS`, not on the streaming
/// end-tag path. `processEOS` used to discard `leadingTextBeforeToolCall`
/// whenever the parse SUCCEEDED, cutting the assistant's visible sentence
/// mid-word on essentially every prose-then-tool-call turn. Live signature
/// (ornith-1.0-9b-mxfp8, temp 0): "…removing the temporary director" with
/// "y." lost, "…finalize the Pipe function in the right plac" from the
/// user report.
struct ProseBeforeToolCallEOSTests {

    private func runCommandTool() -> [String: any Sendable] {
        [
            "type": "function",
            "function": [
                "name": "run_command",
                "description": "Run a shell command.",
                "parameters": [
                    "type": "object",
                    "properties": ["command": ["type": "string"]],
                    "required": ["command"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }

    /// Byte-faithful chunk shapes from the live trace: the prose tail shares
    /// a chunk with the envelope's `<`, and the end tag never fully arrives
    /// before EOS (`</too` held by an upstream holdback).
    @Test("prose survives when the envelope completes at EOS")
    func proseSurvivesEOSCompletion() {
        let processor = ToolCallProcessor(format: .xmlFunction, tools: [runCommandTool()])
        let chunks = [
            "Let me clean up the smoke test artifacts by removing the temporary direct",
            "ory.<to",
            "ol_call>\n<function=run_command>\n<parameter=command>\nrm -rf /tmp/smoke\n",
            "</parameter>\n</function>\n</too",
        ]
        var visible = ""
        for chunk in chunks {
            if let text = processor.processChunk(chunk) {
                visible += text
            }
        }
        if let tail = processor.processEOS() {
            visible += tail
        }

        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "run_command")
        #expect(
            visible
                == "Let me clean up the smoke test artifacts by removing the temporary directory.",
            "the prose before the envelope must be delivered in full")
    }

    @Test("prose survives when the end tag completes mid-stream")
    func proseSurvivesStreamingCompletion() {
        let processor = ToolCallProcessor(format: .xmlFunction, tools: [runCommandTool()])
        let chunks = [
            "Cleaning the directory now.<tool_call>\n<function=run_command>\n",
            "<parameter=command>\nrm -rf /tmp/smoke\n</parameter>\n</function>\n</tool_call>",
        ]
        var visible = ""
        for chunk in chunks {
            if let text = processor.processChunk(chunk) {
                visible += text
            }
        }
        if let tail = processor.processEOS() {
            visible += tail
        }

        #expect(processor.toolCalls.count == 1)
        #expect(visible == "Cleaning the directory now.")
    }

    /// The whole envelope plus its leading prose can arrive in ONE chunk at
    /// EOS (short calls, aggressive coalescing upstream).
    @Test("prose survives a single-chunk envelope at EOS")
    func proseSurvivesSingleChunkEOS() {
        let processor = ToolCallProcessor(format: .xmlFunction, tools: [runCommandTool()])
        var visible = ""
        if let text = processor.processChunk(
            "Done thinking.<tool_call>\n<function=run_command>\n"
                + "<parameter=command>\nls\n</parameter>\n</function>\n</too")
        {
            visible += text
        }
        if let tail = processor.processEOS() {
            visible += tail
        }
        #expect(processor.toolCalls.count == 1)
        #expect(visible == "Done thinking.")
    }

    /// Failed parses keep today's behaviour: the buffered envelope text
    /// flushes back as visible prose together with the leading text.
    @Test("unparsed envelope still flushes with its leading text")
    func unparsedEnvelopeStillFlushes() {
        let processor = ToolCallProcessor(format: .xmlFunction, tools: [runCommandTool()])
        var visible = ""
        if let text = processor.processChunk("Prose head <tool_call>not really a call") {
            visible += text
        }
        if let tail = processor.processEOS() {
            visible += tail
        }
        #expect(processor.toolCalls.isEmpty)
        #expect(visible.hasPrefix("Prose head "))
    }
}
