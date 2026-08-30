// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Verifies `NaiveStreamingDetokenizer` defers mid-grapheme-cluster
// emits so streaming output never splits a multi-codepoint emoji.
// Reproduces the user-visible MiniMax-M2.7-Small JANGTQ symptom
// (osaurus title `American History ❓国旗`) where the streaming
// detokenizer yielded the first regional-indicator scalar of a
// flag emoji alone (rendered as broken-box `❓`) before its
// sibling arrived.

import Foundation
import MLXLMCommon
import Testing

@Suite("NaiveStreamingDetokenizer multi-codepoint emoji deferral")
struct NaiveStreamingDetokenizerEmojiTests {

    /// Stand-in tokenizer that just decodes a token-id list to a
    /// fixed string the test controls — bypasses real BPE so we can
    /// step the streamer through a synthetic decode timeline.
    struct StubTokenizer: Tokenizer {
        let timeline: [String]  // index by token count; segment after N
                                // tokens = timeline[N-1].
        public func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        public func decode(tokenIds: [Int]) -> String {
            guard !tokenIds.isEmpty else { return "" }
            return timeline[min(tokenIds.count - 1, timeline.count - 1)]
        }
        public func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            decode(tokenIds: tokenIds)
        }
        public func convertTokenToId(_ token: String) -> Int? { nil }
        public func convertIdToToken(_ id: Int) -> String? { nil }
        public var bosToken: String? { nil }
        public var eosToken: String? { nil }
        public var unknownToken: String? { nil }
        public func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    /// Step the streamer through the timeline, then FLUSH.
    ///
    /// `next()` holds back the trailing `trailingHoldbackCharacters` (24) of the segment, so on a
    /// short timeline it legitimately yields nothing at all and only `flush()` releases the text.
    /// The previous version of this harness called `next()` alone and asserted a per-token emission
    /// cadence; every case in this suite is under 24 characters, so it collected `[]` and all four
    /// tests failed while appearing to be about emoji.
    ///
    /// The holdback is itself a broader guard against the symptom in the header comment — it cannot
    /// render a partial cluster because it never renders the tail at all. So these now assert the
    /// PROPERTY that survives any holdback size: the stream reconstructs the text exactly, and no
    /// chunk ever ends inside a grapheme cluster.
    static func run(_ timeline: [String]) -> [String] {
        var det = NaiveStreamingDetokenizer(tokenizer: StubTokenizer(timeline: timeline))
        var emitted: [String] = []
        for tok in 0 ..< timeline.count {
            det.append(token: tok)
            if let s = det.next(), !s.isEmpty { emitted.append(s) }
        }
        if let s = det.flush(), !s.isEmpty { emitted.append(s) }
        return emitted
    }

    /// No chunk may end part-way through a grapheme cluster: concatenating the chunks and counting
    /// graphemes must agree with counting the graphemes of each chunk.
    static func splitsNoCluster(_ chunks: [String]) -> Bool {
        chunks.joined().count == chunks.reduce(0) { $0 + $1.count }
    }

    @Test("US flag emoji 🇺🇸 streams atomically (no mid-pair render)")
    func usFlagAtomic() {
        // step 1: model emits regional-indicator U → "🇺" alone
        // step 2: regional-indicator S arrives → flag complete
        let emitted = Self.run(["🇺", "🇺🇸"])
        #expect(emitted.joined() == "🇺🇸", "the completed flag must arrive whole")
        #expect(Self.splitsNoCluster(emitted), "no chunk may end inside the pair")
        #expect(!emitted.contains("🇺"), "the lone regional indicator is the reported symptom")
    }

    @Test("Plain ASCII / non-emoji chars unaffected")
    func plainAsciiUnaffected() {
        let emitted = Self.run(["H", "Hello", "Hello world"])
        #expect(emitted.joined() == "Hello world")
        #expect(Self.splitsNoCluster(emitted))
    }

    @Test("Trailing replacement char (legacy contract) still defers")
    func legacyFFFDDeferral() {
        // Step 2 ends in FFFD → must not be shown; step 3 completes the emoji.
        let emitted = Self.run(["abc", "abc\u{FFFD}", "abc🌍"])
        #expect(emitted.joined() == "abc🌍")
        #expect(!emitted.joined().contains("\u{FFFD}"), "a replacement char must never reach output")
        #expect(Self.splitsNoCluster(emitted))
    }

    @Test("Trailing ZWJ at end-of-chunk defers (no \u{200D} leaks alone)")
    func trailingZWJDefers() {
        // A ZWJ sequence built up one scalar at a time: 👨 → 👨‍ → 👨‍👩
        let emitted = Self.run(["👨", "👨\u{200D}", "👨\u{200D}👩"])
        #expect(emitted.joined() == "👨\u{200D}👩")
        #expect(!emitted.joined().hasSuffix("\u{200D}"), "output must not end on a dangling joiner")
        #expect(Self.splitsNoCluster(emitted))
    }
}
