// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing
import VMLXHub

@testable import VMLXTokenizers

/// BPE works on Unicode scalars and bytes, as Hugging Face tokenizers does, not on grapheme clusters.
///
/// Three faults set it apart from transformers. Seeding the merges from `Character`s kept each
/// cluster whole, so a vocabulary token that starts or ends inside a cluster was never produced.
/// Joining the pieces on " " and splitting them again fused a piece that starts with a combining mark
/// to the space before it, and cut or dropped a piece containing U+0020. And merge keys compared as
/// Swift strings, which are canonical, so canonically equivalent twins shared one rank.
///
/// Each case is a small vocabulary and merge list that shows its fault. The expected ids are what
/// tokenizers 0.22.2 produces for the same model, printed by `scripts/bpe-unicode-scalars-oracle.py`.
/// Ids decide every case, because Swift's `String ==` is canonical too; tokens only label a failure.
struct BPETokenizerUnicodeScalarTests {
    struct Case: CustomTestStringConvertible, Sendable {
        let name: String
        let text: String
        /// Tokens in id order.
        let vocab: [String]
        /// Merges in rank order, as tokenizer.json writes them.
        let merges: [[String]]
        let tokens: [String]
        let ids: [Int]

        var testDescription: String { name }
    }

    static let cases: [Case] = [
        // Granite R2's किसानों: transformers gives किस|ानों, but ानों starts with the vowel sign
        // U+093E, which is inside the cluster सा.
        Case(
            name: "devanagari", text: "किसानों",
            vocab: ["क", "ि", "स", "ा", "न", "ो", "ं", "कि", "किस", "ान", "ानो", "ानों"],
            merges: [["क", "ि"], ["कि", "स"], ["ा", "न"], ["ान", "ो"], ["ानो", "ं"]],
            tokens: ["किस", "ानों"], ids: [8, 11]),
        // "\r\n" is one Character.
        Case(
            name: "crlf", text: "\r\n", vocab: ["\r", "\n"], merges: [],
            tokens: ["\r", "\n"], ids: [0, 1]),
        // 👍🏽: the skin-tone modifier extends the thumbs-up's cluster.
        Case(
            name: "skin-toned emoji", text: "\u{1F44D}\u{1F3FD}",
            vocab: ["\u{1F44D}", "\u{1F3FD}"], merges: [],
            tokens: ["\u{1F44D}", "\u{1F3FD}"], ids: [0, 1]),
        // Spanish té in NFD. The merge (t, e) outranks (e, U+0301), so transformers leaves the accent
        // on its own; the cluster é hides its e from that merge. Escaped because an editor that
        // normalizes to NFC would change the case.
        Case(
            name: "nfd accent", text: "te\u{301}",
            vocab: ["t", "e", "\u{301}", "te", "e\u{301}"],
            merges: [["t", "e"], ["e", "\u{301}"]],
            tokens: ["te", "\u{301}"], ids: [3, 2]),
        // Upstream swift-transformers #355's regression input, which llama-7b tokenizes as its four
        // scalars. The vowel mark U+0E31 is inside the cluster วั.
        Case(
            name: "thai", text: "สวัส", vocab: ["ส", "ว", "ั"], merges: [],
            tokens: ["ส", "ว", "ั", "ส"], ids: [0, 1, 2, 0]),
        // 👍🏽 with the skin-tone modifier outside the vocabulary: only the modifier falls back to its
        // four UTF-8 bytes, where the whole cluster fell back to eight.
        Case(
            name: "byte fallback", text: "\u{1F44D}\u{1F3FD}",
            vocab: ["\u{1F44D}", "<0xF0>", "<0x9F>", "<0x8F>", "<0xBD>"], merges: [],
            tokens: ["\u{1F44D}", "<0xF0>", "<0x9F>", "<0x8F>", "<0xBD>"], ids: [0, 1, 2, 3, 4]),
        // French été in NFC. The merge (é, t) outranks (t, é), but its NFD twin (e + U+0301, t) ranks
        // last: with canonical merge keys the twin's rank replaced the original's, and (t, é) won.
        // Escaped because an editor that normalizes would change the case.
        Case(
            name: "merge keys", text: "\u{E9}t\u{E9}",
            vocab: ["\u{E9}", "t", "\u{E9}t", "t\u{E9}", "e\u{301}", "e\u{301}t"],
            merges: [["\u{E9}", "t"], ["t", "\u{E9}"], ["e\u{301}", "t"]],
            tokens: ["\u{E9}t", "\u{E9}"], ids: [2, 0]),
        // A piece that is only U+0020: splitting the joined pieces on " " dropped it.
        Case(
            name: "space", text: " ", vocab: [" "], merges: [],
            tokens: [" "], ids: [0]),
        // A piece containing U+0020: the split cut it in two.
        Case(
            name: "inner space", text: "a b", vocab: ["a", " ", "b", "a ", "a b"],
            merges: [["a", " "], ["a ", "b"]],
            tokens: ["a b"], ids: [4]),
    ]

    @Test(arguments: cases)
    func tokenizesLikeTransformers(_ testCase: Case) throws {
        let model: [NSString: Any] = [
            "type": "BPE",
            "vocab": Dictionary(
                uniqueKeysWithValues: testCase.vocab.enumerated().map { ($1 as NSString, $0) }),
            "merges": testCase.merges,
            "byte_fallback": true,
        ]
        let tokenizer = try BPETokenizer(
            tokenizerConfig: Config([:] as [NSString: Any]),
            tokenizerData: Config(["model": model] as [NSString: Any]),
            addedTokens: [:])

        let tokens = tokenizer.tokenize(text: testCase.text)
        // nil for a token outside the vocabulary, such as a byte-fallback token.
        let ids = tokens.map { tokenizer.convertTokenToId($0) }
        #expect(
            ids == testCase.ids.map(Optional.some),
            "got \(Self.shown(tokens)), transformers gives \(Self.shown(testCase.tokens))")
    }

    /// Tokens as the oracle prints them: printable ASCII as is, anything else as code points, which
    /// tell NFC from NFD.
    static func shown(_ tokens: [String]) -> String {
        tokens.map { token in
            let scalars = token.unicodeScalars
            if scalars.allSatisfy({ (0x21 ... 0x7E).contains($0.value) }) { return token }
            return scalars.map { String(format: "U+%04X", $0.value) }.joined(separator: " ")
        }.joined(separator: " | ")
    }
}
