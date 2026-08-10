import Foundation
import Testing

@testable import MLXLMCommon

/// Pinned against the LIVE transcript from the first coherent Muse run: raw
/// channel markers were streaming into chat and the `to=self` thinking showed
/// as the answer. The envelope is
/// `<|start|>assistant to=self<|message|>…<|eom|><|start|>assistant to=user<|message|>…<|eot|>`.
@Suite("Muse Glimmer reasoning parser")
struct MuseGlimmerReasoningParserTests {

    func segments(_ raw: String, chunk: Int = 7) -> (reasoning: String, content: String) {
        var parser = ReasoningParser.fromCapabilityName("muse_glimmer")!
        var reasoning = "", content = ""
        var idx = raw.startIndex
        while idx < raw.endIndex {
            let end = raw.index(idx, offsetBy: chunk, limitedBy: raw.endIndex) ?? raw.endIndex
            for seg in parser.feed(String(raw[idx..<end])) {
                switch seg {
                case .reasoning(let r): reasoning += r
                case .content(let c): content += c
                }
            }
            idx = end
        }
        for seg in parser.flush() {
            switch seg {
            case .reasoning(let r): reasoning += r
            case .content(let c): content += c
            }
        }
        return (reasoning, content)
    }

    @Test("the live transcript splits into thinking and answer with no leaks")
    func liveTranscriptSplits() {
        // Condensed from the actual run, structure preserved exactly.
        let raw = "to=user<|message|>to=self<|message|>The user asks a general CS question. "
            + "We should offer to create an agent.<|eom|><|start|>assistant "
            + "to=user<|message|>I can't answer general CS questions directly - "
            + "want me to create a study agent?"

        let (reasoning, content) = segments(raw)

        #expect(reasoning.contains("general CS question"))
        #expect(content.contains("create a study agent"))
        // The failure being pinned: markers must never reach either pane.
        for marker in ["to=self", "to=user", "<|message|>", "<|eom|>", "<|start|>"] {
            #expect(!content.contains(marker), "content leaked \(marker)")
        }
        #expect(!content.contains("The user asks"), "thinking leaked into the answer")
    }

    @Test("a turn with no thinking channel is pure content")
    func noThinkingTurn() {
        let raw = "to=user<|message|>Nearly-sorted input needs few shifts."
        let (reasoning, content) = segments(raw)
        #expect(reasoning.isEmpty)
        #expect(content.contains("few shifts"))
        #expect(!content.contains("to=user"))
    }

    @Test("two thinking blocks both route to reasoning")
    func doubleThinking() {
        let raw = "to=self<|message|>first thought<|eom|><|start|>assistant "
            + "to=self<|message|>second thought<|eom|><|start|>assistant "
            + "to=user<|message|>final answer"
        let (reasoning, content) = segments(raw)
        #expect(reasoning.contains("first thought"))
        #expect(reasoning.contains("second thought"))
        #expect(content.contains("final answer"))
        #expect(!content.contains("<|"))
    }
}
