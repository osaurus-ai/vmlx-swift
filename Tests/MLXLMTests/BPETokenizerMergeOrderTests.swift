// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing
import VMLXHub

@testable import VMLXTokenizers

/// BPE merges in Hugging Face tokenizers' order: the adjacent pair with the lowest rank merges
/// first, the leftmost of equals, and a pair that a merge creates competes at once.
///
/// The order decides the output where a merged pair outranks its own components, as SentencePiece
/// tables rank whitespace runs: Gemma 4's, which Granite Embedding R2 shares, has
/// rank(\t\t, \t) < rank(\t, \t), and Hugging Face splits 32 tabs into 31 + 1.
///
/// Each case is a small vocabulary and merge list. In all but the last, a control, merging every
/// occurrence of a pair before admitting the pairs those merges create gives other tokens. The
/// expected ids are what Hugging Face tokenizers 0.22.2 gives for the same model, with no
/// normalizer or pre-tokenizer:
///
///     Tokenizer(BPE(vocab={token: id, ...}, merges=[(a, b), ...])).encode(text).ids
struct BPETokenizerMergeOrderTests {
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
        // (\t\t, \t) outranks (\t, \t): the pair the first merge creates takes the third tab before
        // (\t, \t) can merge it with the fourth.
        Case(
            name: "tab run", text: "\t\t\t\t",
            vocab: ["\t", "\t\t", "\t\t\t"],
            merges: [["\t\t", "\t"], ["\t", "\t"]],
            tokens: ["\t\t\t", "\t"], ids: [2, 0]),
        // SentencePiece's space, U+2581. Not the longest token first: (▁▁, ▁) takes the third ▁
        // before a second ▁▁ exists to pair with the first.
        Case(
            name: "space run", text: "▁▁▁▁▁",
            vocab: ["▁", "▁▁", "▁▁▁", "▁▁▁▁"],
            merges: [["▁▁", "▁▁"], ["▁▁", "▁"], ["▁", "▁"]],
            tokens: ["▁▁▁", "▁▁"], ids: [2, 1]),
        // A merge admits the pair with its left neighbour too: (a, bb) forms on the left of the
        // first merge, and the (abb, b) it creates takes the third b before (b, b) can merge it
        // with the fourth.
        Case(
            name: "left neighbour", text: "abbbb",
            vocab: ["a", "b", "bb", "abb", "abbb"],
            merges: [["abb", "b"], ["a", "bb"], ["b", "b"]],
            tokens: ["abbb", "b"], ids: [4, 1]),
        // Control: merges in the order BPE training learns them, where every pair ranks after the
        // merges that formed its parts. (\t, \t) merges both occurrences first.
        Case(
            name: "training order", text: "\t\t\t\t",
            vocab: ["\t", "\t\t", "\t\t\t"],
            merges: [["\t", "\t"], ["\t\t", "\t"]],
            tokens: ["\t\t", "\t\t"], ids: [1, 1]),
    ]

    @Test(arguments: cases)
    func mergesInHuggingFaceOrder(_ testCase: Case) throws {
        let model: [NSString: Any] = [
            "type": "BPE",
            "vocab": Dictionary(
                uniqueKeysWithValues: testCase.vocab.enumerated().map { ($1 as NSString, $0) }),
            "merges": testCase.merges,
        ]
        let tokenizer = try BPETokenizer(
            tokenizerConfig: Config([:] as [NSString: Any]),
            tokenizerData: Config(["model": model] as [NSString: Any]),
            addedTokens: [:])

        let tokens = tokenizer.tokenize(text: testCase.text)
        let ids = tokens.map { tokenizer.convertTokenToId($0) }
        #expect(
            ids == testCase.ids.map(Optional.some),
            "got \(tokens), Hugging Face gives \(testCase.tokens)")
    }
}
