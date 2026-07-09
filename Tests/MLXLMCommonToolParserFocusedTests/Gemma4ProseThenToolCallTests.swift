import Foundation
import Testing

@testable import MLXLMCommon

/// Repro for the live E2B-qat leak: gemma-4 emits visible prose, THEN a complete
/// `<|tool_call>call:...{...}<tool_call|>` envelope in the same turn. Observed
/// live (osaurus GUI, gemma-4-E2B-it-qat-MXFP4): the entire well-formed tool call
/// landed in the visible message and never became a tool_calls entry.
///
/// The envelope is spec-correct per gemma-4's chat_template.jinja
/// (`<|tool_call>call:name{key:<|"|>value<|"|>}<tool_call|>`), so extraction
/// must succeed regardless of the leading prose.
@Suite("Gemma-4 prose-then-toolcall is extracted, not leaked")
struct Gemma4ProseThenToolCallTests {

    private func chunked(_ s: String, size: Int) -> [String] {
        var r: [String] = []
        var i = s.startIndex
        while i < s.endIndex {
            let j = s.index(i, offsetBy: size, limitedBy: s.endIndex) ?? s.endIndex
            r.append(String(s[i..<j]))
            i = j
        }
        return r
    }

    private func fileWriteTool() -> [String: any Sendable] {
        [
            "type": "function",
            "function": [
                "name": "file_write",
                "description": "Write text to a file.",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "content": ["type": "string"],
                        "path": ["type": "string"],
                    ],
                    "required": ["content", "path"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }

    // Exact bytes captured from the live leak (osaurus chat-history DB, seq 16).
    private let stream =
        "$12 \\times 8$ is 96. I am creating `e2b_proof.txt` now with the result.\n\n"
        + "<|tool_call>call:file_write{content:<|\"|>RESULT=96<|\"|>,path:<|\"|>e2b_proof.txt<|\"|>}<tool_call|>"

    // Exact reasoning-channel text captured from the same live turn (thinking field).
    private let reasoning =
        "The user wants me to perform the calculation $12 \\times 8$ and then create a new file "
        + "named `e2b_proof.txt` containing the result.\nThe calculation is $12 \\times 8 = 96$.\n"
        + "I need to use the `file_write` tool to create the file.\n\n"
        + "1.  Calculate the result: $12 \\times 8 = 96$.\n2.  Create `e2b_proof.txt` with `RESULT=96`."
    private let contentAfterReasoning =
        "$12 \\times 8$ is 96. I am creating `e2b_proof.txt` now with the result.\n\n"
        + "<|tool_call>call:file_write{content:<|\"|>RESULT=96<|\"|>,path:<|\"|>e2b_proof.txt<|\"|>}<tool_call|>"

    /// Faithful reproduction of the live pump: the SAME processor first sees the
    /// reasoning channel (routed via `routeGenerationText(channel: .reasoning)`),
    /// then the content channel. This is the path the BatchEngine actually runs.
    @Test("reasoning-then-content through the shared processor: tool call still extracted")
    func reasoningThenContentSharedProcessor() {
        for size in [1, 2, 3, 4, 8, 16, 64, 1000] {
            let proc = ToolCallProcessor(format: .gemma4, tools: [fileWriteTool()])
            var visible = ""
            var toolCalls = 0
            func route(_ text: String, _ channel: GenerationTextChannel) {
                for c in chunked(text, size: size) {
                    for ev in routeGenerationText(c, channel: channel, through: proc) {
                        switch ev {
                        case .chunk(let v): visible += v
                        case .toolCall: toolCalls += 1
                        default: break
                        }
                    }
                }
            }
            route(reasoning, .reasoning)
            route(contentAfterReasoning, .content)
            for ev in flushGenerationText(channel: .content, through: proc) {
                switch ev {
                case .chunk(let v): visible += v
                case .toolCall: toolCalls += 1
                default: break
                }
            }
            #expect(
                toolCalls == 1,
                "size=\(size): expected 1 tool call after reasoning→content, got \(toolCalls)")
            #expect(
                !visible.contains("<|tool_call>") && !visible.contains("call:file_write")
                    && !visible.contains("<|\"|>"),
                "size=\(size): tool-call markup leaked into visible content: \(visible.debugDescription)")
        }
    }

    /// FULL pump through the real ReasoningParser: reconstruct the raw stream
    /// (harmony channel-wrapped reasoning + content + native tool call) exactly
    /// as gemma-4 emits it, then run the engine's pump order (parser.feed →
    /// route reasoning, collect content, route content) on a shared processor.
    @Test("full pump: harmony reasoning envelope then content tool call")
    func fullPumpHarmonyThenToolCall() {
        let raw =
            "<|channel>thought\n" + reasoning + "\n<channel|>" + contentAfterReasoning
        for size in [1, 2, 3, 4, 8, 16, 64, 1000] {
            guard var rParser = ReasoningParser.forPrompt(stampName: "harmony", promptTail: nil)
            else { Issue.record("no harmony parser"); return }
            let proc = ToolCallProcessor(format: .gemma4, tools: [fileWriteTool()])
            var visible = ""
            var reasoningOut = ""
            var toolCalls = 0
            func handle(_ events: [Generation]) {
                for ev in events {
                    switch ev {
                    case .chunk(let v): visible += v
                    case .reasoning(let r): reasoningOut += r
                    case .toolCall: toolCalls += 1
                    default: break
                    }
                }
            }
            for chunk in chunked(raw, size: size) {
                var kept: [String] = []
                for seg in rParser.feed(chunk) {
                    switch seg {
                    case .content(let c): kept.append(c)
                    case .reasoning(let r):
                        handle(routeGenerationText(r, channel: .reasoning, through: proc))
                    }
                }
                for piece in kept {
                    handle(routeGenerationText(piece, channel: .content, through: proc))
                }
            }
            for seg in rParser.flush() {
                switch seg {
                case .content(let c):
                    handle(routeGenerationText(c, channel: .content, through: proc))
                case .reasoning(let r):
                    handle(routeGenerationText(r, channel: .reasoning, through: proc))
                }
            }
            handle(flushGenerationText(channel: .content, through: proc))

            #expect(
                toolCalls == 1,
                "size=\(size): expected 1 tool call from full pump, got \(toolCalls) | visible=\(visible.debugDescription)")
            #expect(
                !visible.contains("<|tool_call>") && !visible.contains("call:file_write")
                    && !visible.contains("<|\"|>"),
                "size=\(size): tool-call markup LEAKED into visible content: \(visible.debugDescription)")
        }
    }

    // Exact CONTENT bytes captured from a live E2B-qat PARSER-LOSS leak
    // (quant_leak_sweep [5.1]): the prose contains a stray `<` in
    // `DOUBLED=<product*2>` BEFORE a spec-correct `<|tool_call>` envelope.
    // Live, this call was NOT extracted (tool_calls empty) and the whole
    // envelope leaked into the visible bubble — the false `<` tag-start
    // desynced tag detection so the real start tag was missed.
    private let strayAngleThenToolCall =
        "The previous step added `RESULT=product_placeholder` to `mt_5.txt`. "
        + "Now, I will append the line `DOUBLED=<product*2>` to the file.\n\n"
        + "<|tool_call>call:file_write{content:<|\"|>DOUBLED=product*2<|\"|>,"
        + "mode:<|\"|>append<|\"|>,path:<|\"|>mt_5.txt<|\"|>}<tool_call|>"

    @Test("stray `<` in prose before a gemma4 tool call: call still extracted, no leak")
    func strayAngleBeforeToolCallStillExtracted() {
        for size in [1, 2, 3, 4, 8, 16, 64, 1000] {
            let proc = ToolCallProcessor(format: .gemma4, tools: [fileWriteTool()])
            var visible = ""
            for c in chunked(strayAngleThenToolCall, size: size) {
                if let v = proc.processChunk(c) { visible += v }
            }
            if let tail = proc.processEOS() { visible += tail }

            #expect(
                proc.toolCalls.count == 1,
                "size=\(size): expected 1 tool call, got \(proc.toolCalls.count) | visible=\(visible.debugDescription)")
            #expect(
                proc.toolCalls.first?.function.name == "file_write",
                "size=\(size): wrong/no function name")
            #expect(
                !visible.contains("<|tool_call>") && !visible.contains("<tool_call|>")
                    && !visible.contains("call:file_write") && !visible.contains("<|\"|>"),
                "size=\(size): tool-call markup leaked into visible text: \(visible.debugDescription)")
            // the harmless stray-angle prose must survive as visible text
            #expect(
                visible.contains("DOUBLED=<product*2>"),
                "size=\(size): stray-angle prose was lost: \(visible.debugDescription)")
        }
    }

    // Exact CONTENT bytes from live E2B-qat Mode-B leaks that have NO stray `<`
    // before the envelope (quant_leak_sweep 1338 [19.1], [21.1]). Used to decide
    // whether these represent a second parser trigger or merely a stale binary.
    private let noStrayAngleLeak19 =
        "**Reasoning:**\nTo fulfill the request, I will first calculate the doubled "
        + "value of the previous result (96). Doubling 96 gives 192. Then, I will use "
        + "the `file_write` tool with the append mode to add the new line `DOUBLED=192` "
        + "to the `mt_19.txt` file.\n\n"
        + "<|tool_call>call:file_write{content:<|\"|>DOUBLED=192<|\"|>,"
        + "mode:<|\"|>append<|\"|>,path:<|\"|>mt_19.txt<|\"|>}<tool_call|>"

    private let noStrayAngleLeak21 =
        "**Brief Reasoning:**\nThe multiplication $12 \\times 8$ is calculated by "
        + "breaking down the multiplication into smaller, manageable parts. Since $8$ "
        + "is $2+2+2+2$, we multiply $12$ by each of these components ($12 \\times 2$ "
        + "four times), which results in $24 + 24 + 24 + 24 = 96$.\n\n"
        + "Now, I will append the result to the file.\n\n"
        + "<|tool_call>call:file_write{content:<|\"|>DOUBLED=96<|\"|>,"
        + "mode:<|\"|>append<|\"|>,path:<|\"|>mt_21.txt<|\"|>}<tool_call|>"

    @Test("no-stray-angle live Mode-B content extracts cleanly (stale-binary probe)")
    func noStrayAngleModeBExtracted() {
        for (label, stream) in [("19", noStrayAngleLeak19), ("21", noStrayAngleLeak21)] {
            for size in [1, 2, 3, 4, 8, 16, 64, 1000] {
                let proc = ToolCallProcessor(format: .gemma4, tools: [fileWriteTool()])
                var visible = ""
                for c in chunked(stream, size: size) {
                    if let v = proc.processChunk(c) { visible += v }
                }
                if let tail = proc.processEOS() { visible += tail }
                #expect(
                    proc.toolCalls.count == 1,
                    "leak\(label) size=\(size): expected 1 tool call, got \(proc.toolCalls.count) | visible=\(visible.debugDescription)")
                #expect(
                    !visible.contains("<|tool_call>") && !visible.contains("call:file_write")
                        && !visible.contains("<|\"|>"),
                    "leak\(label) size=\(size): markup leaked: \(visible.debugDescription)")
            }
        }
    }

    /// Isolates the SECOND live mechanism: a stray `<` in the REASONING channel
    /// (gemma-4 echoes the user's `DOUBLED=<product*2>` inside its harmony
    /// thought) is processed through the SAME ToolCallProcessor as content. If
    /// that leaves the processor mid-buffer, the real `<|tool_call>` arriving on
    /// the CONTENT channel afterwards is mis-handled and leaks — even though the
    /// content itself has no stray `<`. Full pump through the real harmony parser.
    @Test("stray `<` in the reasoning channel must not corrupt the content tool call")
    func strayAngleInReasoningDoesNotCorruptContentCall() {
        let reasoningWithStrayAngle =
            "The user asks to append `DOUBLED=<product*2>` to the file. Since "
            + "`<product>` is the previous result 96, `<product*2>` is 192. I will "
            + "use `file_write` with `mode=\"append\"`."
        let contentCall =
            "Now I will append the line.\n\n"
            + "<|tool_call>call:file_write{content:<|\"|>DOUBLED=192<|\"|>,"
            + "mode:<|\"|>append<|\"|>,path:<|\"|>mt_x.txt<|\"|>}<tool_call|>"
        let raw = "<|channel>thought\n" + reasoningWithStrayAngle + "\n<channel|>" + contentCall

        for size in [1, 2, 3, 4, 8, 16, 64, 1000] {
            guard var rParser = ReasoningParser.forPrompt(stampName: "harmony", promptTail: nil)
            else { Issue.record("no harmony parser"); return }
            let proc = ToolCallProcessor(format: .gemma4, tools: [fileWriteTool()])
            var visible = ""
            var toolCalls = 0
            func handle(_ events: [Generation]) {
                for ev in events {
                    switch ev {
                    case .chunk(let v): visible += v
                    case .toolCall: toolCalls += 1
                    default: break
                    }
                }
            }
            for chunk in chunked(raw, size: size) {
                for seg in rParser.feed(chunk) {
                    switch seg {
                    case .content(let c):
                        handle(routeGenerationText(c, channel: .content, through: proc))
                    case .reasoning(let r):
                        handle(routeGenerationText(r, channel: .reasoning, through: proc))
                    }
                }
            }
            for seg in rParser.flush() {
                switch seg {
                case .content(let c):
                    handle(routeGenerationText(c, channel: .content, through: proc))
                case .reasoning(let r):
                    handle(routeGenerationText(r, channel: .reasoning, through: proc))
                }
            }
            handle(flushGenerationText(channel: .content, through: proc))

            #expect(
                toolCalls == 1,
                "size=\(size): expected 1 tool call, got \(toolCalls) | visible=\(visible.debugDescription)")
            #expect(
                !visible.contains("<|tool_call>") && !visible.contains("call:file_write")
                    && !visible.contains("<|\"|>"),
                "size=\(size): tool-call markup leaked into content: \(visible.debugDescription)")
        }
    }

    @Test("prose then a complete gemma4 tool call: call extracted, no markup leaked")
    func proseThenToolCallExtracted() {
        for size in [1, 2, 3, 4, 8, 16, 64, 1000] {
            let proc = ToolCallProcessor(format: .gemma4, tools: [fileWriteTool()])
            var visible = ""
            for c in chunked(stream, size: size) {
                if let v = proc.processChunk(c) { visible += v }
            }
            if let tail = proc.processEOS() { visible += tail }

            // 1) the tool call must be extracted
            #expect(
                proc.toolCalls.count == 1,
                "size=\(size): expected 1 tool call, got \(proc.toolCalls.count)")
            #expect(
                proc.toolCalls.first?.function.name == "file_write",
                "size=\(size): wrong/no function name")
            // 2) no raw tool-call markup may leak into visible text
            #expect(
                !visible.contains("<|tool_call>") && !visible.contains("<tool_call|>")
                    && !visible.contains("call:file_write") && !visible.contains("<|\"|>"),
                "size=\(size): tool-call markup leaked into visible text: \(visible.debugDescription)")
            // 3) the leading prose must survive
            #expect(
                visible.contains("is 96") && visible.contains("e2b_proof.txt"),
                "size=\(size): leading prose was lost: \(visible.debugDescription)")
        }
    }
}
