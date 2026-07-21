// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Streaming text-level stop-sequence matcher.
// See `Libraries/MLXLMCommon/BatchEngine/STOP-SEQUENCES-CONTRACT.md`.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("StopStringMatcher")
struct StopStringMatcherTests {
    private struct NoSpecialTokenTokenizer: Tokenizer {
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    private struct UnknownMappingTokenizer: Tokenizer {
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { "<unk>" }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { 0 }
        func convertIdToToken(_ id: Int) -> String? { id == 0 ? "<unk>" : nil }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    // MARK: - Disabled matcher (no stop strings)

    @Test("empty stop list → pass-through")
    func testEmptyIsPassThrough() {
        var m = StopStringMatcher(stopStrings: [])
        #expect(!m.isEnabled)
        switch m.feed("hello world") {
        case .streaming(let out):
            #expect(out == "hello world")
        case .stopped:
            Issue.record("Should not stop when no strings are configured")
        }
        #expect(m.flush() == "")
    }

    @Test("all-empty-string input is filtered, matcher disabled")
    func testFiltersEmptyStrings() {
        var m = StopStringMatcher(stopStrings: ["", "", ""])
        #expect(!m.isEnabled)
        if case .streaming(let out) = m.feed("anything") {
            #expect(out == "anything")
        } else {
            Issue.record("Disabled matcher should always stream")
        }
    }

    @Test("duplicates are de-duplicated at init")
    func testDeduplicates() {
        let m = StopStringMatcher(stopStrings: ["END", "END", "STOP", "END"])
        #expect(m.stopStrings.count == 2)
        #expect(m.stopStrings == ["END", "STOP"])
    }

    // MARK: - Single feed, single stop string

    @Test("stop string in the middle of a single feed splits correctly")
    func testSingleFeedMiddleMatch() {
        var m = StopStringMatcher(stopStrings: ["STOP"])
        switch m.feed("answer STOP ignored tail") {
        case .stopped(let out):
            #expect(out == "answer ")
        case .streaming:
            Issue.record("Expected a stop match")
        }
    }

    @Test("stop string at start of single feed — empty emit")
    func testSingleFeedStartMatch() {
        var m = StopStringMatcher(stopStrings: ["STOP"])
        switch m.feed("STOP trailing") {
        case .stopped(let out):
            #expect(out.isEmpty)
        case .streaming:
            Issue.record("Expected stop match at offset 0")
        }
    }

    @Test("stop string at end of single feed — full emit")
    func testSingleFeedEndMatch() {
        var m = StopStringMatcher(stopStrings: ["END"])
        switch m.feed("some text END") {
        case .stopped(let out):
            #expect(out == "some text ")
        case .streaming:
            Issue.record("Expected stop match")
        }
    }

    // MARK: - Streaming across chunks

    @Test("stop string split across two chunks")
    func testSplitAcrossChunks() {
        var m = StopStringMatcher(stopStrings: ["STOP"])

        // First feed contains a partial match — the last (maxStopLen-1 == 3)
        // characters "ST " are held as potential prefix, so emit is
        // everything before them ("before").
        let r1 = m.feed("before ST")
        switch r1 {
        case .streaming(let out):
            #expect(out == "before")
        case .stopped:
            Issue.record("Not yet a full match")
        }

        // Second feed completes the match. The buffer at this point is
        // " ST" + "OP after" = " STOP after" — match lands at offset 1,
        // so the pre-match emit is the single space.
        let r2 = m.feed("OP after")
        switch r2 {
        case .stopped(let out):
            #expect(out == " ")
        case .streaming:
            Issue.record("Expected stopped on second chunk")
        }
    }

    @Test("char-by-char streaming holds all bytes until disambiguated")
    func testCharByCharStreaming() {
        var m = StopStringMatcher(stopStrings: ["</end>"])
        var emitted = ""
        var stopped = false

        // The matcher holds up to 5 chars (maxStopLen-1 = 5) waiting.
        for ch in "hello</end>tail" {
            let res = m.feed(String(ch))
            switch res {
            case .streaming(let out):
                emitted += out
            case .stopped(let out):
                emitted += out
                stopped = true
                break
            }
            if stopped { break }
        }
        #expect(stopped)
        // Only "hello" is safely emittable — everything after "hello" is
        // part of or preceding the match.
        #expect(emitted == "hello")
    }

    @Test("flush drains held tail when no match arrives")
    func testFlushDrainsTail() {
        var m = StopStringMatcher(stopStrings: ["STOP"])
        // "partial match potential" doesn't contain STOP; 3 chars at the
        // end ("ial") are held as potential prefix.
        switch m.feed("partial match potential") {
        case .streaming(let out):
            // Emitted all but last (maxStopLen - 1 = 3) characters
            #expect(out == "partial match potent")
        case .stopped:
            Issue.record("No stop string present")
        }
        // Flush releases the held tail.
        #expect(m.flush() == "ial")
    }

    // MARK: - Multiple stop strings

    @Test("earliest match across multiple stop strings wins")
    func testEarliestMatchAcrossMultiple() {
        var m = StopStringMatcher(stopStrings: ["ENDING", "STOP", "FIN"])
        // "FIN" appears first (offset 9 vs STOP at 13) — earliest wins.
        switch m.feed("something FIN then STOP") {
        case .stopped(let out):
            #expect(out == "something ")
        case .streaming:
            Issue.record("Expected a stop match")
        }
    }

    @Test("different-length stop strings respect longest hold size")
    func testDifferentLengthHold() {
        var m = StopStringMatcher(stopStrings: ["X", "LONGSTOP"])
        // After feeding "partial LONGSTO", the last 7 chars ("LONGSTO")
        // are potentially a prefix of "LONGSTOP" — must be held.
        let r = m.feed("partial LONGSTO")
        switch r {
        case .streaming(let out):
            #expect(out == "partial ")
        case .stopped:
            Issue.record("No full match yet")
        }
        // Complete match → stop.
        switch m.feed("P rest") {
        case .stopped(let out):
            #expect(out.isEmpty)
        case .streaming:
            Issue.record("Expected stop")
        }
    }

    // MARK: - No match, natural termination

    @Test("stream finishes with no match → flush returns all held text")
    func testNaturalTerminationFlushesHeld() {
        var m = StopStringMatcher(stopStrings: ["STOP"])
        _ = m.feed("a") // holds "a"
        _ = m.feed("b") // holds "ab"
        // No more text. Flush.
        let tail = m.flush()
        #expect(tail == "ab")
    }

    @Test("repeated feeds then flush reassemble the original text")
    func testRoundTripByFlush() {
        var m = StopStringMatcher(stopStrings: ["SOMETHING_THAT_NEVER_APPEARS"])
        var emitted = ""
        let input = "the quick brown fox jumps over the lazy dog"
        for ch in input {
            switch m.feed(String(ch)) {
            case .streaming(let out):
                emitted += out
            case .stopped:
                Issue.record("Stop string not in text")
            }
        }
        emitted += m.flush()
        #expect(emitted == input)
    }

    // MARK: - GenerateParameters plumbing

    @Test("GenerateParameters.extraStopStrings round-trips through init")
    func testGenerateParametersField() {
        let params = GenerateParameters(extraStopStrings: ["END", "STOP"])
        #expect(params.extraStopStrings == ["END", "STOP"])

        let defaults = GenerateParameters()
        #expect(defaults.extraStopStrings.isEmpty)
    }

    @Test("unresolved special-token text becomes default stop strings")
    func testUnresolvedSpecialTokenTextFallbacks() {
        let resolved = resolveStopSequences(
            modelConfiguration: ModelConfiguration(id: "nemotron-h"),
            tokenizer: NoSpecialTokenTokenizer(),
            includeUnknownToken: true)

        #expect(resolved.tokenIDs.isEmpty)
        #expect(resolved.textStopStrings.contains("<｜end▁of▁sentence｜>"))
        #expect(resolved.textStopStrings.contains("<｜end▁of▁sentence>"))
        #expect(resolved.textStopStrings.contains("<｜end▁of▁sent"))
        #expect(!resolved.textStopStrings.contains("<|im_end|>"))
    }

    @Test("unknown-token mappings still fall back to narrow sentinel text")
    func testUnknownTokenMappingUsesSentinelTextFallbacks() {
        let resolved = resolveStopSequences(
            modelConfiguration: ModelConfiguration(id: "nemotron-h"),
            tokenizer: UnknownMappingTokenizer(),
            includeUnknownToken: true)

        #expect(resolved.tokenIDs == [0])
        #expect(resolved.textStopStrings.contains("<｜end▁of▁sentence｜>"))
        #expect(resolved.textStopStrings.contains("<｜end▁of▁sentence>"))
        #expect(resolved.textStopStrings.contains("<｜end▁of▁sent"))
        #expect(!resolved.textStopStrings.contains("<|im_end|>"))
    }

    @Test("explicit unresolved extra EOS token becomes exact text fallback")
    func testExplicitUnresolvedExtraEOSTokenTextFallback() {
        let resolved = resolveStopSequences(
            modelConfiguration: ModelConfiguration(
                id: "custom",
                extraEOSTokens: ["<|custom_end|>"]),
            tokenizer: NoSpecialTokenTokenizer())

        #expect(resolved.textStopStrings.contains("<|custom_end|>"))
    }

    @Test("default sentinel fallback truncates Nemotron malformed text")
    func testDefaultSentinelFallbackTruncatesMalformedText() {
        let resolved = resolveStopSequences(
            modelConfiguration: ModelConfiguration(id: "nemotron-h"),
            tokenizer: NoSpecialTokenTokenizer())
        var matcher = StopStringMatcher(stopStrings: resolved.textStopStrings)

        switch matcher.feed("Your word is blue.<｜end▁of▁sentic>") {
        case .stopped(let out):
            #expect(out == "Your word is blue.")
        case .streaming:
            Issue.record("Expected malformed sentinel text to stop generation")
        }

        var matcher2 = StopStringMatcher(stopStrings: resolved.textStopStrings)
        switch matcher2.feed("Finish with blue.<｜end▁of▁sentence>") {
        case .stopped(let out):
            #expect(out == "Finish with blue.")
        case .streaming:
            Issue.record("Expected missing-final-pipe sentinel text to stop generation")
        }
    }

    @Test("default sentinel fallback does not truncate normal prose")
    func testDefaultSentinelFallbackDoesNotTruncateNormalProse() {
        let resolved = resolveStopSequences(
            modelConfiguration: ModelConfiguration(id: "nemotron-h"),
            tokenizer: NoSpecialTokenTokenizer())
        var matcher = StopStringMatcher(stopStrings: resolved.textStopStrings)

        let input = "Use the phrase end of sentence in normal prose without stopping."
        var emitted = ""
        for ch in input {
            switch matcher.feed(String(ch)) {
            case .streaming(let out):
                emitted += out
            case .stopped:
                Issue.record("Default sentinel fallback should not stop on ordinary prose")
            }
        }

        emitted += matcher.flush()
        #expect(emitted == input)
    }

    // MARK: - generation_config.json eos is authoritative (harmony / gpt-oss)

    /// The motivating case, proven rather than asserted. gpt-oss (OpenAI harmony) declares
    /// `eos_token_id: [200002 <|return|>, 199999, 200012 <|call|>]` and deliberately EXCLUDES
    /// 200007 `<|end|>`, which is a CHANNEL SEPARATOR, not a stop: the model emits
    /// `<|end|>` to close the `analysis` channel before opening `final`. The hard-coded
    /// `commonEndTokenStrings` blanket used to inject `<|end|>` anyway, so generation halted at
    /// the end of the reasoning channel and the visible answer was empty.
    @Test("declared generation_config eos suppresses the blanket (harmony <|end|> is not a stop)")
    func harmonyDeclaredEOSKeepsChannelSeparatorOutOfStops() throws {
        let harmonyGenerationConfig = """
        { "eos_token_id": [200002, 199999, 200012], "do_sample": true, "temperature": 1.0 }
        """
        let defaults = try JSONDecoder.json5().decode(
            GenerationConfigFile.self, from: Data(harmonyGenerationConfig.utf8))
        let declaredEOS = try #require(defaults.eosTokenIds?.values)
        #expect(declaredEOS == [200002, 199999, 200012])
        #expect(!declaredEOS.contains(200007))

        let resolved = resolveStopSequences(
            modelConfiguration: ModelConfiguration(id: "gpt-oss-20b", generationDefaults: defaults),
            tokenizer: HarmonyTokenizer())

        // 200007 <|end|> must NOT become a stop: the declaration is authoritative.
        #expect(!resolved.tokenIDs.contains(200007))
        #expect(!resolved.textStopStrings.contains("<|end|>"))
        // ...while the model's own declared stops are honored.
        #expect(resolved.tokenIDs.contains(200002))
    }

    /// The other side of the gate — the failure mode that would actually hurt. A model that
    /// declares its eos must STILL stop; the gate only suppresses the hard-coded blanket, it
    /// must never suppress the declaration itself or the tokenizer eos.
    @Test("a declared-eos model still stops (Qwen <|im_end|>)")
    func declaredEOSModelStillStops() throws {
        let qwenGenerationConfig = """
        { "eos_token_id": [151645, 151643] }
        """
        let defaults = try JSONDecoder.json5().decode(
            GenerationConfigFile.self, from: Data(qwenGenerationConfig.utf8))

        let resolved = resolveStopSequences(
            modelConfiguration: ModelConfiguration(
                id: "qwen3", extraEOSTokens: ["<|im_end|>"], generationDefaults: defaults),
            tokenizer: QwenImEndTokenizer())

        // <|im_end|> resolves (id 151645 via the tokenizer) and the declared ids are kept.
        #expect(resolved.tokenIDs.contains(151645))
        #expect(!resolved.tokenIDs.isEmpty)
    }

    /// Tokenizer stub: knows harmony's specials, so <|end|> (200007) IS resolvable —
    /// proving the gate (not an unresolvable token) is what keeps it out of the stop set.
    private struct HarmonyTokenizer: Tokenizer {
        var bosToken: String? { nil }
        var eosToken: String? { "<|return|>" }
        var unknownToken: String? { nil }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? {
            switch token {
            case "<|end|>": return 200007
            case "<|return|>": return 200002
            case "<|call|>": return 200012
            default: return nil
            }
        }
        func convertIdToToken(_ id: Int) -> String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    /// Tokenizer stub for a Qwen-style bundle: <|im_end|> resolves to its real id.
    private struct QwenImEndTokenizer: Tokenizer {
        var bosToken: String? { nil }
        var eosToken: String? { "<|im_end|>" }
        var unknownToken: String? { nil }
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { token == "<|im_end|>" ? 151645 : nil }
        func convertIdToToken(_ id: Int) -> String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

}
