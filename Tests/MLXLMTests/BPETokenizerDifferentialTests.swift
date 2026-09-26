// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Differential + performance regression for the O(n log n) BPE merge rewrite
// (osaurus-ai/vmlx-swift#73). The optimized `BPETokenizer.bpe(token:)` (heap +
// doubly-linked list) must merge in Hugging Face tokenizers' order. A heap
// whose stale entries are skipped lazily is easy to get subtly wrong, so that
// order MUST be locked by a test rather than trusted.
//
// This test states the order directly as the reference oracle (merge the
// adjacent pair with the lowest (rank, left index), then rescan: O(n²) and
// plainly correct), builds a real BPETokenizer from an on-disk Gemma merge
// table, and asserts the shipped `bpe()` matches the reference across
// thousands of fuzzed inputs — including the ~11k-char whitespace-free
// pre-token (compact tool JSON) that motivated #73. Skips when no Gemma
// tokenizer is on the machine. BPETokenizerMergeOrderTests checks the order
// against Hugging Face's own output on small tables.

import Foundation
import XCTest

import VMLXHub
@testable import VMLXTokenizers

final class BPETokenizerDifferentialTests: XCTestCase {

    // MARK: reference oracle — Hugging Face tokenizers' merge order, stated directly

    private func referenceBpe(_ token: String, _ bpeRanks: [BytePair: Int]) -> [String] {
        // Hugging Face's Word::merge_all pops the lowest (rank, position) from a heap and admits
        // the pairs each merge creates at once. Rescanning for the lowest (rank, left index) after
        // every merge is the same order without the heap. Like the shipped bpe(), it seeds from
        // Unicode scalars, returns an array, and looks pairs up as BytePairs, which compare
        // literally, since String == is canonical. BPETokenizerUnicodeScalarTests checks the
        // Unicode handling. The results still compare exactly with ==: both segment the same
        // scalars.
        var word = token.unicodeScalars.map { String($0) }
        while word.count > 1 {
            var best: (rank: Int, left: Int)?
            for left in 0 ..< word.count - 1 {
                // Strictly lower, so the leftmost of equal ranks wins.
                if let rank = bpeRanks[BytePair(word[left], word[left + 1])],
                    rank < best?.rank ?? .max
                {
                    best = (rank, left)
                }
            }
            guard let best else { break }
            word[best.left] += word[best.left + 1]
            word.remove(at: best.left + 1)
        }
        return word
    }

    // MARK: deterministic PRNG so failures reproduce exactly

    private struct LCG: RandomNumberGenerator {
        var state: UInt64
        init(seed: UInt64) { state = seed == 0 ? 0x9E3779B97F4A7C15 : seed }
        mutating func next() -> UInt64 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return state
        }
    }

    // MARK: real Gemma BPETokenizer from disk

    private func loadGemmaBPETokenizer() throws -> BPETokenizer? {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let candidates = [
            "/Users/eric/osaurus_models/finished/gemma-4-26b-a4b-it-4bit",
            "/Users/eric/osaurus_models/finished/gemma-4-e4b-it-4bit",
            "/Users/eric/osaurus_models/finished/gemma-4-e2b-it-4bit",
            home + "/MLXModels/OsaurusAI/gemma-4-12B-it-qat-JANG_4M",
        ]
        let fm = FileManager.default
        guard
            let dir = candidates.first(where: {
                fm.fileExists(atPath: $0 + "/tokenizer.json")
            })
        else { return nil }

        func config(_ name: String) throws -> Config {
            let url = URL(fileURLWithPath: dir).appendingPathComponent(name)
            let obj = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
            guard let dict = obj as? [NSString: Any] else {
                throw XCTSkip("\(name) is not a JSON object")
            }
            return Config(dict)
        }

        let tokenizerData = try config("tokenizer.json")
        let tokenizerConfig = try config("tokenizer_config.json")
        return try BPETokenizer(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerData,
            addedTokens: [:])
    }

    private func loadDSV4Tokenizer() throws -> (PreTrainedTokenizer, BPETokenizer)? {
        let directory = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("models/DeepSeek-V4-Flash-0731-JANG")
        guard FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("tokenizer.json").path)
        else { return nil }

        func config(_ name: String) throws -> Config {
            let url = directory.appendingPathComponent(name)
            let obj = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
            guard let dict = obj as? [NSString: Any] else {
                throw XCTSkip("\(name) is not a JSON object")
            }
            return Config(dict)
        }

        let tokenizerData = try config("tokenizer.json")
        let tokenizerConfig = try config("tokenizer_config.json")
        let model = try BPETokenizer(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerData,
            addedTokens: [:])
        let tokenizer = try PreTrainedTokenizer(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerData)
        return (tokenizer, model)
    }

    /// The fuzz alphabet: the merge table's parts that are a single `Character`.
    private func atomicAlphabet(_ bpeRanks: [BytePair: Int]) -> [String] {
        // Literal scalars as keys: a Set<String> folds canonical twins, keeping one by hash order.
        var parts = Set<[Unicode.Scalar]>()
        // Characters, not scalars: a multi-scalar part re-merges from its scalars, so merges fire.
        for bp in bpeRanks.keys {
            if bp.a.count == 1 { parts.insert(Array(bp.a.unicodeScalars)) }
            if bp.b.count == 1 { parts.insert(Array(bp.b.unicodeScalars)) }
        }
        // Sorted by scalar values: the alphabet, and so every fuzzed word, is the same each run.
        return parts.sorted { $0.lexicographicallyPrecedes($1) }
            .map { String(String.UnicodeScalarView($0)) }
    }

    private func randomWord(
        _ rng: inout LCG, alphabet: [String], length: Int
    ) -> String {
        var s = ""
        s.reserveCapacity(length * 2)
        for _ in 0..<length {
            s += alphabet[Int(rng.next() % UInt64(alphabet.count))]
        }
        return s
    }

    // MARK: tests

    /// The shipped optimized `bpe()` must merge in Hugging Face's order on every
    /// input, over the REAL Gemma merge table.
    func testOptimizedBpeMatchesReferenceAcrossFuzzedInputs() throws {
        guard let tok = try loadGemmaBPETokenizer() else {
            throw XCTSkip("No Gemma tokenizer on this machine.")
        }
        let ranks = tok.bpeRanks
        XCTAssertGreaterThan(ranks.count, 1000, "merge table looks empty")
        let alphabet = atomicAlphabet(ranks)
        XCTAssertGreaterThan(alphabet.count, 10, "alphabet looks empty")

        var rng = LCG(seed: 0xB9E_CAFE_1234)
        var checked = 0

        // Many short/medium words — the common case + merge-rank-tie stress
        // (repeated single chars exercise the left-to-right non-overlap order).
        for _ in 0..<4000 {
            let len = 2 + Int(rng.next() % 40)
            let w = randomWord(&rng, alphabet: alphabet, length: len)
            XCTAssertEqual(tok.bpe(token: w), referenceBpe(w, ranks),
                "divergence on input (len \(len)): \(w.debugDescription)")
            checked += 1
        }

        // Adversarial repeated-character runs (e.g. "aaaa…") which maximize
        // same-rank ties — the case most likely to expose ordering bugs.
        for ch in alphabet.prefix(12) {
            for len in [2, 3, 4, 5, 8, 16, 33, 64, 129] {
                let w = String(repeating: ch, count: len)
                XCTAssertEqual(tok.bpe(token: w), referenceBpe(w, ranks),
                    "divergence on repeated \(ch.debugDescription)×\(len)")
                checked += 1
            }
        }

        // Longer words (hundreds–thousands of symbols).
        for _ in 0..<40 {
            let len = 200 + Int(rng.next() % 2000)
            let w = randomWord(&rng, alphabet: alphabet, length: len)
            XCTAssertEqual(tok.bpe(token: w), referenceBpe(w, ranks),
                "divergence on long input (len \(len))")
            checked += 1
        }

        print("[BPEDifferential] \(checked) inputs byte-identical against reference")
    }

    /// The pathological case from the PR: one ~11k-char whitespace-free
    /// pre-token. Assert (a) the optimized output still matches the reference,
    /// and (b) it runs fast (the original took ~6 s; the optimized ~tens of ms).
    func testLongSpaceFreePreTokenIsCorrectAndFast() throws {
        guard let tok = try loadGemmaBPETokenizer() else {
            throw XCTSkip("No Gemma tokenizer on this machine.")
        }
        let ranks = tok.bpeRanks
        let alphabet = atomicAlphabet(ranks)
        var rng = LCG(seed: 0x11035_BEEF)
        let big = randomWord(&rng, alphabet: alphabet, length: 11_035)

        let start = Date()
        let optimized = tok.bpe(token: big)
        let elapsed = Date().timeIntervalSince(start)

        XCTAssertEqual(optimized, referenceBpe(big, ranks),
            "optimized bpe diverged on the 11k-char pre-token")
        // Generous ceiling — the original was multiple seconds; the optimized
        // path should be well under 1s even on CI.
        XCTAssertLessThan(elapsed, 1.0,
            "optimized bpe took \(elapsed)s on an 11k-char pre-token — the "
            + "quadratic may have regressed")
        print("[BPEDifferential] 11k-char pre-token: optimized bpe in "
            + String(format: "%.1f ms", elapsed * 1000))
    }

    /// DSV4's tokenizer deliberately keeps punctuation plus trailing newlines
    /// in one pre-token. The canonical Rust/Python tokenizer therefore merges
    /// `.\n\n` to vocab token 339 (`.ĊĊ`). Losing that boundary changes every
    /// native DSML prompt and was observed live as corrupted tool names.
    func testDSV4PunctuationNewlinePreTokenMatchesCanonicalIDs() throws {
        guard let (tokenizer, model) = try loadDSV4Tokenizer() else {
            throw XCTSkip("Local DSV4 0731 tokenizer is unavailable.")
        }
        XCTAssertEqual(model.bpe(token: ".ĊĊ"), referenceBpe(".ĊĊ", model.bpeRanks))
        XCTAssertEqual(model.bpe(token: ".ĊĊ"), [".ĊĊ"])
        XCTAssertEqual(tokenizer.encode(text: ".\n\n", addSpecialTokens: false), [339])
        XCTAssertEqual(tokenizer.tokenize(text: ".\n\n"), [".ĊĊ"])
    }

    /// JSON decoding materializes `\\r` / `\\n` regex escapes as control
    /// scalars. Foundation needs them re-escaped to retain Hugging Face's
    /// greedy punctuation-plus-newline pre-token boundary.
    func testJSONDecodedControlScalarsKeepCanonicalPreTokenBoundary() {
        let rawPattern =
            #" ?[\p{P}\p{S}]+["# + "\r\n" + #"]*|\s*["# + "\r\n" + #"]+|\s+"#
        let config = Config([
            "type": "Split",
            "pattern": ["Regex": rawPattern],
            "behavior": "Isolated",
            "invert": false,
        ] as [NSString: Any])
        let tokenizer = SplitPreTokenizer(config: config)

        XCTAssertEqual(
            foundationCompatibleTokenizerRegex(rawPattern),
            #" ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+"#)
        XCTAssertEqual(tokenizer.preTokenize(text: ".\n\n", options: []), [".\n\n"])
    }
}
