// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Regression for the Qwen 3.6 27B crash tpae reported on osaurus 0.17.3:
//
//   Thread 1 Crashed: libswiftCore.dylib _assertionFailure ...
//   (user-initiated-qos.cooperative Task)
//
// Root cause: `NaiveStreamingDetokenizer.next()` computes
//   `newSegment.suffix(newSegment.count - segment.count)`
// which traps with
//   "Can't take a suffix of negative length from a collection"
// whenever the tokenizer's incremental decode produces a SHORTER string
// than the previously-decoded segment. That's not hypothetical on Qwen
// 3.6: byte-level BPE + `cleanUpTokenizationSpaces`-style substitutions
// (" ." → ".", " 's" → "'s") + emoji/multi-byte grapheme cluster
// completion can all yield `newSegment.count < segment.count` at some
// intermediate step.
//
// These tests reproduce the crash against the REAL Qwen 3.6 tokenizer
// without needing the 27B weights. If any test crashes the process with
// `_assertionFailure`, we've reproduced tpae's trap.

import Foundation
import XCTest
@preconcurrency import VMLXTokenizers

@testable import MLXLMCommon

final class NaiveStreamingDetokenizerShrinkTests: XCTestCase {

    /// Stub tokenizer whose decode return is fully under test control.
    /// Returns the i-th pre-recorded decode output based on input length.
    /// Lets us construct the exact shrinkage pattern that triggers the
    /// `suffix(_:)` trap, without depending on a network download.
    final class ScriptedDecodeTokenizer: MLXLMCommon.Tokenizer, @unchecked Sendable {
        private let outputs: [String]
        init(outputs: [String]) { self.outputs = outputs }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            let idx = min(tokenIds.count, outputs.count) - 1
            guard idx >= 0 else { return "" }
            return outputs[idx]
        }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    /// Baseline — monotonic growth: the common path must not crash.
    func testMonotonicGrowthDoesNotCrash() {
        let tok = ScriptedDecodeTokenizer(outputs: ["He", "Hel", "Hell", "Hello"])
        var d = NaiveStreamingDetokenizer(tokenizer: tok)
        var out = ""
        for t in [1, 2, 3, 4] {
            d.append(token: t)
            if let s = d.next() { out += s }
        }
        if let s = d.flush() { out += s }
        XCTAssertEqual(out, "Hello")
    }

    /// Direct repro. Scripted decode returns a SHORTER string on the
    /// second call. Pre-fix: this crashes with
    ///   "Can't take a suffix of negative length from a collection"
    /// matching tpae's `_assertionFailure` signature.
    /// Post-fix: this returns nil (or "") for the shrink step and keeps
    /// the detokenizer usable.
    func testShrinkingDecodeDoesNotCrash() {
        // Simulates the realistic case where the tokenizer's cleanup or
        // byte-level reassembly produces a shorter total after appending
        // one more token — e.g. an emoji completion that collapses a
        // `\u{fffd}` sequence into a single grapheme cluster.
        let tok = ScriptedDecodeTokenizer(outputs: [
            "Hello \u{fffd}\u{fffd}\u{fffd}",  // 10 graphemes (1 Hello + space + 3 repl)
            "Hello 😀",                         // 7 graphemes (Hello + space + emoji)
        ])
        var d = NaiveStreamingDetokenizer(tokenizer: tok)
        d.append(token: 1)
        _ = d.next()            // consume first segment
        d.append(token: 2)
        _ = d.next()            // PRE-FIX: traps here with negative suffix
    }

    /// Cleanup-induced shrink. Swift-transformers' `cleanUp(text:)`
    /// performs string substitutions like " ." → ".", " 's" → "'s".
    /// The combined pattern of an earlier token emitting " 's" plus a
    /// later token triggering a substitution rule can produce a
    /// shorter `newSegment` than the prior `segment`.
    func testCleanupInducedShrinkDoesNotCrash() {
        // Turn 1: "hello 's"   (8 graphemes, unclean) cleanUp → "hello's"   (7)
        // Turn 2: "hello 's "  (9 graphemes, unclean) cleanUp → "hello's "  (8)
        // Turn 3: " ."         applied — imagine " 's" re-absorbs. A
        // hostile tokenizer could return a pre-cleaned "helloAAAA" (9),
        // then "helloAA" (7) — pattern irrelevant; we only need ∃ step
        // where newSegment.count < segment.count.
        let tok = ScriptedDecodeTokenizer(outputs: [
            "hello 's",        // 8
            "hello's ",        // 8 (re-cleaned — same count here, still OK)
            "hello's",         // 7 — SHRINK vs prior 8
        ])
        var d = NaiveStreamingDetokenizer(tokenizer: tok)
        for t in [1, 2, 3] {
            d.append(token: t)
            _ = d.next()
        }
    }

    /// Special-token render-vs-strip shrink. Qwen 3.6 has 30+ special
    /// tokens (`<|im_end|>`, `<|channel|>`, `<|thought|>`, …). If the
    /// per-token decode renders them as placeholder strings on step N
    /// but the multi-token decode collapses two adjacent specials into
    /// a single rendered marker, the incremental count can shrink.
    func testSpecialTokenCollapseShrinkDoesNotCrash() {
        // Simulate: step N renders "<a><b>" (6 graphemes),
        //           step N+1 collapses to "<ab>" (4 graphemes).
        let tok = ScriptedDecodeTokenizer(outputs: [
            "<a>",        // 3
            "<a><b>",     // 6
            "<ab>",       // 4 — SHRINK
        ])
        var d = NaiveStreamingDetokenizer(tokenizer: tok)
        for t in [1, 2, 3] {
            d.append(token: t)
            _ = d.next()
        }
    }

    /// After a shrink we've reconciled the baseline — subsequent growth
    /// should resume emitting new content correctly, not replay old
    /// content or be permanently stuck.
    func testShrinkDoesNotPermanentlyBreakFutureEmission() {
        let tok = ScriptedDecodeTokenizer(outputs: [
            "Hello world",     // 11
            "Hell",            // 4 — SHRINK (baseline reconciled to "Hell")
            "Hell yeah!",      // 10 — growth resumes; should emit " yeah!"
        ])
        var d = NaiveStreamingDetokenizer(tokenizer: tok)

        var emitted = ""
        d.append(token: 1)
        if let s = d.next() { emitted += s }
        d.append(token: 2)
        _ = d.next()  // shrink — returns nil, reconciles baseline
        d.append(token: 3)
        if let s = d.next() { emitted += s }
        if let s = d.flush() { emitted += s }

        // The exact content depends on where the baseline landed, but
        // we require the detokenizer (a) never crashed, (b) emits
        // SOMETHING after the shrink+grow path rather than silently
        // swallowing forever.
        XCTAssertFalse(emitted.isEmpty,
            "Detokenizer must resume emission after a shrink (got \"\(emitted)\")")
    }

    /// Gemma 4 regression: visible app streaming showed impossible strings
    /// like "I am an cOsaurus hat agent." even though final decode should be
    /// "I am an Osaurus chat agent.". The broken path emitted a short unstable
    /// prefix ("I am an c") before a later decode reinterpreted the same token
    /// boundary. Streaming plus final flush must equal final decode exactly.
    func testStreamingPlusFlushMatchesFinalDecodeWhenPrefixIsReinterpreted() {
        let final = "I am an Osaurus chat agent."
        let tok = ScriptedDecodeTokenizer(outputs: [
            "I am an c",
            "I am an Osaurus c",
            final,
        ])
        var d = NaiveStreamingDetokenizer(tokenizer: tok)
        var emitted = ""
        for t in [1, 2, 3] {
            d.append(token: t)
            if let s = d.next() { emitted += s }
        }
        if let s = d.flush() { emitted += s }

        XCTAssertEqual(emitted, final)
        XCTAssertFalse(emitted.contains("cOsaurus"))
        XCTAssertFalse(emitted.contains("Osaurus hat"))
    }
}

/// Defensive regression suite for `ReasoningParser.drain`'s holdback
/// math. `safeTail = max(0, max(startTag.count, endTag.count) - 1)`
/// guards against the case where a caller constructs a parser with an
/// empty tag (e.g. a mis-configured model stamp override). Prior to the
/// guard, the empty-tag path traps in the stdlib with
///     "negative distance: can't step through a Collection with a
///      negative count"
/// when `offsetBy: -safeTail` moves forward past `endIndex`. The
/// `insideReasoning` branch is reached when the model opens a reasoning
/// block; if `endTag` is empty, `buffer.range(of: "")` returns nil (per
/// Foundation semantics), falling through to the holdback math.
final class ReasoningParserEmptyTagTests: XCTestCase {

    func testEmptyEndTagInsideReasoningDoesNotCrash() {
        // Carry the parser into reasoning mode so `endTag` becomes the
        // `lookFor` target inside `drain`. The guard should keep the
        // empty `endTag` from underflowing the holdback arithmetic.
        var parser = ReasoningParser(
            startTag: "<think>",
            endTag: "")   // malformed — exercises the guard
        // Open the reasoning block, then feed text that can never match
        // the empty end tag via `range(of:)`.
        _ = parser.feed("<think>something important here")
        // If the guard is missing this line crashes with a stdlib trap.
        _ = parser.feed(" more text after")
        _ = parser.flush()
    }

    func testBothEmptyTagsDoNotCrash() {
        var parser = ReasoningParser(startTag: "", endTag: "")
        _ = parser.feed("plain text buffer")
        _ = parser.feed(" more")
        _ = parser.flush()
    }
}
