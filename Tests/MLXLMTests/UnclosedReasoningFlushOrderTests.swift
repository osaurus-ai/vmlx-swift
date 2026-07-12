// `GenerateCompletionInfo.unclosedReasoning` reports the "trapped thinking"
// pathology: the stream ended while the model was still inside a reasoning
// block, so the answer is stuck in the reasoning pane. Consumers surface it
// as a warning (osaurus renders "⚠ thinking didn't close — answer may be in
// reasoning above").
//
// It was reported for Hunyuan v3 on streams that closed perfectly cleanly —
// `reasoning_content` and `content` came back correctly separated with
// `finish_reason=stop`, and the warning still fired.
//
// The cause is an ordering bug, not a parser bug. The streaming detokenizer
// withholds a trailing window of characters (`trailingHoldbackCharacters`)
// so it never emits a partial UTF-8 grapheme; that held text only reaches
// the reasoning parser when the generation loop flushes at EOS. Both engines
// read `isInsideReasoning` BEFORE that flush — so any close marker short
// enough to still be sitting in the holdback had not been parsed yet, and
// the flag came back true.
//
// `</think:opensource>` is 20 characters against a 24-character holdback, so
// it hides completely. `</think>` is 8 and clears it — which is why the whole
// think_xml family looked fine and only Hunyuan v3 misreported.
//
// Reading the flag after the flush is not a fix either: `ReasoningParser.flush()`
// ends by setting `insideReasoning = false` unconditionally, which would report
// "closed" even for a model that genuinely stopped mid-thought — erasing the
// signal this flag exists for. The state has to be captured in the window
// between the two.
//
// Source-coverage style — no MLX runtime needed.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("unclosedReasoning is captured after the detokenizer flush")
struct UnclosedReasoningFlushOrderTests {

    private static func source(_ relativePath: String) throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MLXLMTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // repo root
        let url =
            root
            .appendingPathComponent("Libraries/MLXLMCommon")
            .appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// The precondition for the bug: Hunyuan v3's close marker is short enough
    /// to sit entirely inside the detokenizer's held-back tail, so at the moment
    /// the loop ends it has not been parsed yet.
    @Test("Hy3's close marker fits inside the detokenizer holdback")
    func closeMarkerFitsInsideHoldback() {
        let hy3Close = "</think:opensource>"
        #expect(hy3Close.count <= NaiveStreamingDetokenizer.trailingHoldbackCharacters)
        // The plain marker clears the holdback, which is why the rest of the
        // think_xml family never surfaced this.
        #expect("</think>".count < NaiveStreamingDetokenizer.trailingHoldbackCharacters)
    }

    /// The mechanism, on the real parser: a stream whose close marker has not
    /// yet been pumped through still reads as "inside reasoning". Once the held
    /// tail arrives — exactly what the engines' end-of-stream flush does — the
    /// parser closes. So the flag must be read after that pump, not before.
    @Test("Parser reads as unclosed until the held-back tail is pumped through")
    func heldBackCloseMarkerFlipsTheFlag() {
        var parser = ReasoningParser(
            startTag: "<think:opensource>",
            endTag: "</think:opensource>",
            startInReasoning: true)

        // Everything the detokenizer has emitted so far: the reasoning body,
        // with the close marker + answer still held in its 24-char tail.
        _ = parser.feed("The user wants an exact string. I will output it.")
        #expect(
            parser.isInsideReasoning,
            "before the held tail arrives the parser is legitimately still inside reasoning")

        // The end-of-stream detokenizer flush delivers the held tail.
        _ = parser.feed("</think:opensource>BANNER CHECK OK")
        #expect(
            !parser.isInsideReasoning,
            "the close marker has now been parsed — the stream did NOT end inside reasoning")
    }

    /// A model that genuinely stops mid-thought must still be reported, so the
    /// capture cannot simply move to after `ReasoningParser.flush()` — that call
    /// clears the flag unconditionally.
    @Test("flush() clears insideReasoning, so it cannot be the capture point")
    func flushClearsTheFlag() {
        var parser = ReasoningParser(
            startTag: "<think:opensource>",
            endTag: "</think:opensource>",
            startInReasoning: true)
        _ = parser.feed("thinking forever with no close tag")
        #expect(parser.isInsideReasoning, "genuinely trapped: no close marker was ever emitted")

        _ = parser.flush()
        #expect(
            !parser.isInsideReasoning,
            "flush() clears the flag — reading here would erase the trapped-thinking signal")
    }

    /// Pin the fixed ordering in both engines: the flag is read after the flush
    /// that pumps the detokenizer's held tail through the parser.
    @Test("Both engines capture the flag after the end-of-stream flush")
    func enginesCaptureAfterFlush() throws {
        let evaluate = try Self.source("Evaluate.swift")
        let flushCall = try #require(evaluate.range(of: "handler.onGenerationEnd(emit: continuation.yield)"))
        let flagRead = try #require(
            evaluate.range(
                of: "let unclosedReasoning = handler.unclosedReasoning",
                range: flushCall.upperBound ..< evaluate.endIndex),
            "Evaluate must read unclosedReasoning AFTER onGenerationEnd flushes the detokenizer tail through the parser"
        )
        #expect(flushCall.upperBound <= flagRead.lowerBound)

        // The handler snapshots the state in the window between the detokenizer
        // flush and the parser flush, and reports that snapshot in preference to
        // the live (already-cleared) parser state.
        #expect(evaluate.contains("terminalInsideReasoning = reasoningParser?.isInsideReasoning ?? false"))
        #expect(
            evaluate.contains(
                "terminalInsideReasoning ?? (reasoningParser?.isInsideReasoning ?? false)"))

        let batch = try Self.source("BatchEngine/BatchEngine.swift")
        #expect(batch.contains("var terminalInsideReasoning: Bool? = nil"))
        #expect(
            batch.contains(
                "terminalInsideReasoning ?? (reasoningParser?.isInsideReasoning ?? false)"))
        #expect(
            !batch.contains("let unclosed = reasoningParser?.isInsideReasoning ?? false"),
            "BatchEngine must not read the live parser state before flush()"
        )
    }
}
