import Foundation
import Testing

@testable import MLXLMCommon

/// Muse Glimmer's turn is a recipient-channel envelope:
///
/// ```
/// <|start|>assistant to=self<|message|>THINKING<|eom|>
/// <|start|>assistant to=user<|message|>ANSWER<|eot|>
/// ```
///
/// A tool call names the tool as the recipient, and that spelling matched no
/// tag or alias, so `to=underwriting_daily_summary<|message|>` streamed
/// verbatim into the reasoning rail — consistently at the end of the first
/// think block, because `<|start|>assistant` had already opened reasoning.
///
/// The header is protocol. These pin that it is consumed, that `self`/`user`
/// keep their tag meanings, and that ordinary prose containing `to=` is not
/// eaten.
@Suite("Muse recipient channel headers")
struct MuseRecipientHeaderTests {

    private func parser() throws -> ReasoningParser {
        try #require(ReasoningParser.fromCapabilityName("muse_glimmer"))
    }

    private func run(_ parser: inout ReasoningParser, _ chunks: [String])
        -> (reasoning: String, content: String)
    {
        var reasoning = "", content = ""
        for chunk in chunks {
            for segment in parser.feed(chunk) {
                switch segment {
                case .reasoning(let t): reasoning += t
                case .content(let t): content += t
                }
            }
        }
        for segment in parser.flush() {
            switch segment {
            case .reasoning(let t): reasoning += t
            case .content(let t): content += t
            }
        }
        return (reasoning, content)
    }

    @Test("a tool recipient header never reaches the rail")
    func toolHeaderIsConsumed() throws {
        var p = try parser()
        let out = run(&p, [
            "<|start|>assistant to=self<|message|>Let's do that.",
            "<|start|>assistant to=underwriting_underwriter_daily_summary<|message|>",
            "{\"since\":\"yesterday\"}<|eom|>",
        ])
        #expect(!out.reasoning.contains("to=underwriting_underwriter_daily_summary"))
        #expect(!out.reasoning.contains("<|message|>"))
        #expect(out.reasoning.contains("Let's do that."))
        #expect(out.content.isEmpty)
    }

    @Test("the header still goes when split across chunks")
    func splitHeaderIsConsumed() throws {
        var p = try parser()
        let out = run(&p, [
            "<|start|>assistant to=self<|message|>thinking",
            "<|start|>assistant to=under",
            "writing_daily<|mess",
            "age|>body<|eom|>",
        ])
        #expect(!out.reasoning.contains("to=under"))
        #expect(!out.reasoning.contains("<|message|>"))
        #expect(out.reasoning.contains("thinking"))
    }

    /// The regression that matters: `to=user<|message|>` is the END alias. If
    /// header consumption swallowed it, reasoning would never close and the
    /// whole answer would stream as thinking.
    @Test("to=user still closes reasoning so the answer is content")
    func userRecipientStillClosesReasoning() throws {
        var p = try parser()
        let out = run(&p, [
            "<|start|>assistant to=self<|message|>deliberating<|eom|>",
            "<|start|>assistant to=user<|message|>Here is the answer.<|eot|>",
        ])
        #expect(out.reasoning.contains("deliberating"))
        #expect(out.content.contains("Here is the answer."))
        #expect(!out.content.contains("to=user"))
    }

    @Test("to=self still opens reasoning")
    func selfRecipientStillOpensReasoning() throws {
        var p = try parser()
        let out = run(&p, ["<|start|>assistant to=self<|message|>private thought<|eom|>"])
        #expect(out.reasoning.contains("private thought"))
        #expect(out.content.isEmpty)
    }

    @Test("a full turn: think, call a tool, then answer")
    func fullTurn() throws {
        var p = try parser()
        let out = run(&p, [
            "<|start|>assistant to=self<|message|>I need yesterday's summary.<|eom|>",
            "<|start|>assistant to=daily_summary<|message|>{\"since\":\"yesterday\"}<|eom|>",
            "<|start|>assistant to=user<|message|>Yesterday had 14 items.<|eot|>",
        ])
        #expect(out.reasoning.contains("I need yesterday's summary."))
        // `<|eot|>` stripping is a separate, pre-existing gap; assert on the
        // answer text rather than exact equality.
        #expect(out.content.contains("Yesterday had 14 items."))
        #expect(!out.content.contains("since"), "the tool-call body leaked into the answer")
        #expect(!out.reasoning.contains("to=daily_summary"))
        #expect(!out.content.contains("<|message|>"))
    }

    // MARK: - Must not eat ordinary text

    @Test("prose containing to= is left alone")
    func proseWithToEqualsSurvives() throws {
        var p = try parser()
        let out = run(&p, [
            "<|start|>assistant to=user<|message|>",
            "Set the query to=all and compare it to=none, then report.<|eot|>",
        ])
        #expect(out.content.contains("to=all"))
        #expect(out.content.contains("to=none"))
    }

    @Test("a recipient name with spaces is prose, not a header")
    func spacedRecipientIsNotAHeader() throws {
        var p = try parser()
        let out = run(&p, [
            "<|start|>assistant to=user<|message|>",
            "compare to=the first value<|message|> and stop.<|eot|>",
        ])
        #expect(out.content.contains("to=the first value"))
    }

    @Test("the classifier accepts identifiers and rejects prose")
    func recipientNameShapes() {
        for good in ["daily_summary", "a", "tool.v2", "web-search", "T9"] {
            #expect(ReasoningParser.isRecipientName(good), "rejected \(good)")
        }
        for bad in ["", "two words", "has\nnewline", String(repeating: "x", count: 129)] {
            #expect(!ReasoningParser.isRecipientName(bad), "accepted \(bad.debugDescription)")
        }
    }

    // MARK: - The shapes that actually ship

    /// Live Muse output ends the header at `<|message|` with no closing `>`,
    /// because the tool-call envelope after it is consumed downstream.
    /// Requiring the full spelling meant the header never matched — this is
    /// the exact text that leaked in production.
    @Test("the open `<|message|` terminator is enough")
    func openTerminatorIsConsumed() throws {
        var p = try parser()
        let out = run(&p, [
            "<|start|>assistant to=self<|message|>Let's call it. ",
            "to=get_current_time<|message|",
        ])
        #expect(!out.reasoning.contains("to=get_current_time"))
        #expect(!out.reasoning.contains("<|message|"))
        #expect(out.reasoning.contains("Let's call it."))
    }

    /// And when the turn simply ends on the header, flush must DROP it rather
    /// than emit what it was holding.
    @Test("a header held at end-of-stream is dropped, not flushed")
    func heldHeaderIsDroppedOnFlush() throws {
        var p = try parser()
        let out = run(&p, [
            "<|start|>assistant to=self<|message|>thinking about it. ",
            "to=some_tool",
        ])
        #expect(!out.reasoning.contains("to=some_tool"))
        #expect(out.reasoning.contains("thinking about it."))
        #expect(out.content.isEmpty)
    }

    @Test("the closing `>` is swallowed when it does arrive")
    func closingBracketIsSwallowed() throws {
        var p = try parser()
        let out = run(&p, [
            "<|start|>assistant to=self<|message|>a. ",
            "to=tool_x<|message|>body<|eom|>",
        ])
        #expect(!out.reasoning.contains(">body"), "leaked the closing bracket")
        #expect(out.reasoning.contains("body"))
    }
}
