import Foundation
import VMLXHub
import VMLXTokenizers
import XCTest

/// CPU-only, hardcoded stock tokenizers 0.22.2 goldens. No weights, installed
/// model, Metal, template rewrite or roundtrip-only equality is involved.
final class SerializedMetaspaceContractTests: XCTestCase {
    private let vocabulary = [
        "<unk>": 0, "▁": 1, "a": 2, "▁▁": 3, "▁▁a": 4, "▁a": 5,
        "<bos>": 6, "<tag>": 7, "\n": 8, "_": 9, "__": 10, "__a": 11,
        "_a": 12, "b": 13, "▁b": 14, "_b": 15,
    ]
    private let merges = [
        ["▁", "▁"], ["▁▁", "a"], ["▁", "a"],
        ["_", "_"], ["__", "a"], ["_", "a"], ["▁", "b"], ["_", "b"],
    ]

    private func tokenizer(
        split: Bool?, scheme: String? = "first", replacement: String = "▁",
        addPrefixSpace: Bool? = nil, vocabulary: [String: Int]? = nil,
        merges: [[String]]? = nil
    ) throws -> PreTrainedTokenizer {
        var pre: [String: Any] = ["type": "Metaspace", "replacement": replacement]
        if let split { pre["split"] = split }
        if let scheme { pre["prepend_scheme"] = scheme }
        if let addPrefixSpace { pre["add_prefix_space"] = addPrefixSpace }
        let single: [[String: Any]] = [
            ["SpecialToken": ["id": "<bos>", "type_id": 0]],
            ["Sequence": ["id": "A", "type_id": 0]],
        ]
        let data: [NSString: Any] = [
            "model": [
                "type": "BPE", "unk_token": "<unk>",
                "vocab": vocabulary ?? self.vocabulary, "merges": merges ?? self.merges,
            ] as [String: Any],
            "pre_tokenizer": pre,
            "added_tokens": [
                ["id": 6, "content": "<bos>", "special": true] as [String: Any],
                ["id": 7, "content": "<tag>", "special": false] as [String: Any],
            ],
            "post_processor": ["type": "TemplateProcessing", "single": single, "pair": single],
        ]
        return try PreTrainedTokenizer(
            tokenizerConfig: Config(["tokenizer_class": "GPT2Tokenizer", "unk_token": "<unk>"] as [NSString: Any]),
            tokenizerData: Config(data))
    }

    func testSerializedFalseAllowsCrossSpaceMerges() throws {
        let tok = try tokenizer(split: false)
        for (text, expected) in [
            ("  a", [4]), ("a  ", [5, 3]), ("a b", [5, 14]),
            ("▁a", [5]), ("\na", [1, 8, 2]),
        ] {
            XCTAssertEqual(tok.encode(text: text, addSpecialTokens: false), expected, text)
        }
    }

    func testSerializedTrueAndMissingFieldRetainSplitDefault() throws {
        for split in [true, nil] as [Bool?] {
            let tok = try tokenizer(split: split)
            XCTAssertEqual(tok.encode(text: "  a", addSpecialTokens: false), [1, 5])
            XCTAssertEqual(tok.encode(text: "a  ", addSpecialTokens: false), [5, 1, 1])
            XCTAssertEqual(tok.encode(text: "a b", addSpecialTokens: false), [5, 14])
        }
    }

    func testActualFiveSpaceTokenShapeIsNotFiveIndependentTokens() throws {
        // This is the exact causal vector observed in native request60/61:
        // stock one465 versus five152343, with every other token unchanged.
        let vocab = ["<unk>": 0, "▁": 152_343, "▁▁": 3, "▁▁▁▁": 4, "▁▁▁▁▁": 465]
        let ranks = [["▁", "▁"], ["▁▁", "▁▁"], ["▁▁▁▁", "▁"]]
        let whole = try tokenizer(split: false, vocabulary: vocab, merges: ranks)
        let separated = try tokenizer(split: true, vocabulary: vocab, merges: ranks)
        XCTAssertEqual(whole.encode(text: "     ", addSpecialTokens: false), [465])
        XCTAssertEqual(separated.encode(text: "     ", addSpecialTokens: false), Array(repeating: 152_343, count: 5))
    }

    func testPrependFirstDoesNotRestartAfterAddedTokens() throws {
        for split in [false, true] {
            let tok = try tokenizer(split: split)
            XCTAssertEqual(tok.encode(text: "a<bos>b", addSpecialTokens: false), [5, 6, 13])
            XCTAssertEqual(tok.encode(text: "<bos>a", addSpecialTokens: false), [6, 2])
            XCTAssertEqual(tok.encode(text: "a<tag>b", addSpecialTokens: false), [5, 7, 13])
        }
    }

    func testAlwaysAndNeverPrependRemainDistinct() throws {
        for split in [false, true] {
            let always = try tokenizer(split: split, scheme: "always")
            let never = try tokenizer(split: split, scheme: "never")
            XCTAssertEqual(always.encode(text: "a<bos>b", addSpecialTokens: false), [5, 6, 14])
            XCTAssertEqual(always.encode(text: "<bos>a", addSpecialTokens: false), [6, 5])
            XCTAssertEqual(never.encode(text: "a<bos>b", addSpecialTokens: false), [2, 6, 13])
            XCTAssertEqual(never.encode(text: "<bos>a", addSpecialTokens: false), [6, 2])
        }
    }

    func testLegacyPrefixOptionFallbackAndExplicitSchemePrecedence() throws {
        let omitted = try tokenizer(split: false, scheme: nil)
        let enabled = try tokenizer(split: false, scheme: nil, addPrefixSpace: true)
        let disabled = try tokenizer(split: false, scheme: nil, addPrefixSpace: false)
        let explicit = try tokenizer(split: false, scheme: "first", addPrefixSpace: false)
        XCTAssertEqual(omitted.encode(text: "a<bos>b", addSpecialTokens: false), [5, 6, 14])
        XCTAssertEqual(enabled.encode(text: "a<bos>b", addSpecialTokens: false), [5, 6, 14])
        XCTAssertEqual(disabled.encode(text: "a<bos>b", addSpecialTokens: false), [2, 6, 13])
        XCTAssertEqual(explicit.encode(text: "a<bos>b", addSpecialTokens: false), [5, 6, 13])
    }

    func testAlternativeReplacementIsPreserved() throws {
        let whole = try tokenizer(split: false, replacement: "_")
        let separated = try tokenizer(split: true, replacement: "_")
        XCTAssertEqual(whole.encode(text: "  a", addSpecialTokens: false), [11])
        XCTAssertEqual(whole.encode(text: "a  ", addSpecialTokens: false), [12, 10])
        XCTAssertEqual(whole.encode(text: "a b", addSpecialTokens: false), [12, 15])
        XCTAssertEqual(whole.encode(text: "▁a", addSpecialTokens: false), [9, 5])
        XCTAssertEqual(separated.encode(text: "  a", addSpecialTokens: false), [9, 12])
        XCTAssertEqual(separated.encode(text: "a  ", addSpecialTokens: false), [12, 9, 9])
    }

    func testSpecialTokenPostprocessingRemainsExplicit() throws {
        let tok = try tokenizer(split: false)
        XCTAssertEqual(tok.encode(text: "a", addSpecialTokens: false), [5])
        XCTAssertEqual(tok.encode(text: "a", addSpecialTokens: true), [6, 5])
        // A template-authored BOS stays exactly one when extras are disabled.
        XCTAssertEqual(tok.encode(text: "<bos>a", addSpecialTokens: false), [6, 2])
    }

    func testEmptyWithoutPrependingRemainsEmpty() throws {
        // The existing empty+first/always prepend behavior is a separate
        // backlog item, not changed or hidden by the serialized split fix.
        for split in [false, true] {
            let tok = try tokenizer(split: split, scheme: "never")
            XCTAssertEqual(tok.encode(text: "", addSpecialTokens: false), [])
            XCTAssertEqual(tok.encode(text: "", addSpecialTokens: true), [6])
        }
    }
}
