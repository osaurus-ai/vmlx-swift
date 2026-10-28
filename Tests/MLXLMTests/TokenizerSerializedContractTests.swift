// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import VMLXHub
import XCTest

@testable import VMLXTokenizers

final class TokenizerSerializedContractTests: XCTestCase {
    private func bpe(ignore: Bool?, vocab: [String], merges: [[String]] = []) throws -> BPETokenizer
    {
        var model: [NSString: Any] = [
            "type": "BPE", "merges": merges,
            "vocab": Dictionary(
                uniqueKeysWithValues: vocab.enumerated().map { ($1 as NSString, $0) }),
        ]
        if let ignore { model["ignore_merges"] = ignore }
        return try BPETokenizer(
            tokenizerConfig: Config([:] as [NSString: Any]),
            tokenizerData: Config(["model": model] as [NSString: Any]), addedTokens: [:])
    }

    func testWholeVocabularyPieceBypassesMergesOnlyWhenDeclared() throws {
        let vocab = ["a", "b", "c", "abc"]
        XCTAssertEqual(try bpe(ignore: true, vocab: vocab).tokenize(text: "abc"), ["abc"])
        XCTAssertEqual(try bpe(ignore: false, vocab: vocab).tokenize(text: "abc"), ["a", "b", "c"])
        XCTAssertEqual(try bpe(ignore: nil, vocab: vocab).tokenize(text: "abc"), ["a", "b", "c"])
    }

    func testIgnoredMergesStillUseTheMergeTableForUnknownWholePieces() throws {
        let model = try bpe(
            ignore: true, vocab: ["a", "b", "c", "d", "ab", "abc"],
            merges: [["a", "b"], ["ab", "c"]])
        XCTAssertEqual(model.tokenize(text: "abcd"), ["abc", "d"])
    }

    func testWholePieceLookupKeepsDistinctUnicodeSpellings() throws {
        let composed = "éx"
        let decomposed = "e\u{301}x"
        let model = try bpe(ignore: true, vocab: ["é", "e", "\u{301}", "x", composed, decomposed])
        XCTAssertEqual(model.tokenize(text: composed).first.flatMap(model.convertTokenToId), 4)
        XCTAssertEqual(model.tokenize(text: decomposed).first.flatMap(model.convertTokenToId), 5)
    }

    func testFlagComesFromTheSerializedModelRatherThanTokenizerOptions() throws {
        let model = try BPETokenizer(
            tokenizerConfig: Config(["ignore_merges": true] as [NSString: Any]),
            tokenizerData: Config(
                [
                    "model": [
                        "type": "BPE", "merges": [],
                        "vocab": ["a": 0, "b": 1, "ab": 2],
                    ]
                ] as [NSString: Any]), addedTokens: [:])
        XCTAssertEqual(model.tokenize(text: "ab"), ["a", "b"])
    }

    func testSerializedRegexPiecesMatchTheCanonicalTokenizer() throws {
        struct Fixture: Decodable {
            let name: String
            let pattern: String
            let text: String
            let pieces: [String]
        }
        let fixtures = try JSONDecoder().decode([Fixture].self, from: Data(Self.fixtures.utf8))
        for fixture in fixtures {
            let config = Config(
                [
                    "type": "Split", "pattern": ["Regex": fixture.pattern],
                    "behavior": "Isolated", "invert": false,
                ] as [NSString: Any])
            XCTAssertEqual(
                SplitPreTokenizer(config: config).preTokenize(text: fixture.text, options: []),
                fixture.pieces, fixture.name)
        }
    }

    func testRegexCapturesKeepTheEntireDelimiter() {
        let config = Config(
            [
                "type": "Split", "pattern": ["Regex": "(ab)(c)"],
                "behavior": "Isolated", "invert": false,
            ] as [NSString: Any])
        XCTAssertEqual(
            SplitPreTokenizer(config: config).preTokenize(text: "abc xyz", options: []),
            ["abc", " xyz"])
    }

    func testInvalidRegexLeavesTheOriginalTextAvailable() {
        let config = Config(
            [
                "type": "Split", "pattern": ["Regex": "["],
                "behavior": "Isolated", "invert": false,
            ] as [NSString: Any])
        XCTAssertEqual(
            SplitPreTokenizer(config: config).preTokenize(text: "unaltered", options: []),
            ["unaltered"])
    }

    func testZeroWidthRegexSplitsDoNotEmitEmptyPieces() {
        for (pattern, expected) in [
            ("", ["a", "b"]), ("^|$", ["ab"]),
            ("(?=b)", ["a", "b"]), ("a*", ["a", "b"]),
        ] {
            let config = Config(
                [
                    "type": "Split", "pattern": ["Regex": pattern],
                    "behavior": "Isolated", "invert": false,
                ] as [NSString: Any])
            let tokenizer = SplitPreTokenizer(config: config)
            XCTAssertEqual(tokenizer.preTokenize(text: "ab", options: []), expected, pattern)
            XCTAssertEqual(tokenizer.preTokenize(text: "", options: []), [], pattern)
        }
    }

    // Whole-match pieces from Hugging Face tokenizers 0.23.3. Six serialized
    // patterns, three whitespace/Unicode/code inputs each. No model files or GPU.
    private static let fixtures = #"""
        [
          {
            "name": "pattern0-corpus0",
            "pattern": "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s",
            "text": "a\r\n\ndef\treturn",
            "pieces": [
              "a",
              "\r\n\n",
              "def",
              "\treturn"
            ]
          },
          {
            "name": "pattern0-corpus1",
            "pattern": "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s",
            "text": "Café 👩‍💻 العربية 1234\n",
            "pieces": [
              "Cafe",
              "́",
              " 👩‍💻",
              " العربية",
              " ",
              "123",
              "4",
              "\n"
            ]
          },
          {
            "name": "pattern0-corpus2",
            "pattern": "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s",
            "text": "abc (def)\n\n",
            "pieces": [
              "abc",
              " (",
              "def",
              ")\n\n"
            ]
          },
          {
            "name": "pattern1-corpus0",
            "pattern": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            "text": "a\r\n\ndef\treturn",
            "pieces": [
              "a",
              "\r\n\n",
              "def",
              "\treturn"
            ]
          },
          {
            "name": "pattern1-corpus1",
            "pattern": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            "text": "Café 👩‍💻 العربية 1234\n",
            "pieces": [
              "Café",
              " 👩‍💻",
              " العربية",
              " ",
              "1",
              "2",
              "3",
              "4",
              "\n"
            ]
          },
          {
            "name": "pattern1-corpus2",
            "pattern": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            "text": "abc (def)\n\n",
            "pieces": [
              "abc",
              " (",
              "def",
              ")\n\n"
            ]
          },
          {
            "name": "pattern2-corpus0",
            "pattern": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            "text": "a\r\n\ndef\treturn",
            "pieces": [
              "a",
              "\r\n\n",
              "def",
              "\treturn"
            ]
          },
          {
            "name": "pattern2-corpus1",
            "pattern": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            "text": "Café 👩‍💻 العربية 1234\n",
            "pieces": [
              "Cafe",
              "́",
              " 👩‍💻",
              " العربية",
              " ",
              "123",
              "4",
              "\n"
            ]
          },
          {
            "name": "pattern2-corpus2",
            "pattern": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            "text": "abc (def)\n\n",
            "pieces": [
              "abc",
              " (",
              "def",
              ")\n\n"
            ]
          },
          {
            "name": "pattern3-corpus0",
            "pattern": "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            "text": "a\r\n\ndef\treturn",
            "pieces": [
              "a",
              "\r\n\n",
              "def",
              "\treturn"
            ]
          },
          {
            "name": "pattern3-corpus1",
            "pattern": "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            "text": "Café 👩‍💻 العربية 1234\n",
            "pieces": [
              "Café",
              " 👩",
              "‍",
              "💻",
              " العربية",
              " ",
              "1234",
              "\n"
            ]
          },
          {
            "name": "pattern3-corpus2",
            "pattern": "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            "text": "abc (def)\n\n",
            "pieces": [
              "abc",
              " (",
              "def",
              ")\n\n"
            ]
          },
          {
            "name": "pattern4-corpus0",
            "pattern": "[一-龥぀-ゟ゠-ヿ]+",
            "text": "a\r\n\ndef\treturn",
            "pieces": [
              "a\r\n\ndef\treturn"
            ]
          },
          {
            "name": "pattern4-corpus1",
            "pattern": "[一-龥぀-ゟ゠-ヿ]+",
            "text": "Café 👩‍💻 العربية 1234\n",
            "pieces": [
              "Café 👩‍💻 العربية 1234\n"
            ]
          },
          {
            "name": "pattern4-corpus2",
            "pattern": "[一-龥぀-ゟ゠-ヿ]+",
            "text": "abc (def)\n\n",
            "pieces": [
              "abc (def)\n\n"
            ]
          },
          {
            "name": "pattern5-corpus0",
            "pattern": "\\p{N}{1,3}",
            "text": "a\r\n\ndef\treturn",
            "pieces": [
              "a\r\n\ndef\treturn"
            ]
          },
          {
            "name": "pattern5-corpus1",
            "pattern": "\\p{N}{1,3}",
            "text": "Café 👩‍💻 العربية 1234\n",
            "pieces": [
              "Café 👩‍💻 العربية ",
              "123",
              "4",
              "\n"
            ]
          },
          {
            "name": "pattern5-corpus2",
            "pattern": "\\p{N}{1,3}",
            "text": "abc (def)\n\n",
            "pieces": [
              "abc (def)\n\n"
            ]
          }
        ]
        """#
}
