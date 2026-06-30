// Offset-aware WordPiece tokenizer for Rampart (BERT uncased).
//
// Faithfully mirrors HuggingFace `BertTokenizer` (BasicTokenizer +
// WordpieceTokenizer) for `do_lower_case=True`: text cleaning, CJK spacing,
// lowercasing, accent stripping, and punctuation splitting — while carrying the
// character span each token covers in the ORIGINAL string, which PII redaction
// needs.
//
// Offsets are character indices into `Array(text)` (matching how `RampartPII`
// slices the text). Normalization that expands a character (e.g. accent
// decomposition or multi-char lowercasing) maps every produced character back
// to its single origin index, so spans stay aligned to the input.

import Foundation

public struct RampartTokenizer: Sendable {
    public struct Token: Sendable {
        public let id: Int
        /// Character-index range into the original text, or nil for [CLS]/[SEP].
        public let range: Range<Int>?
    }

    /// A normalized character tagged with its origin index in the input.
    private struct NormChar {
        let ch: Character
        let origin: Int
    }

    private let vocab: [String: Int]
    private let unkId: Int
    private let clsId: Int
    private let sepId: Int
    private let maxInputCharsPerWord = 100
    public let maxLength: Int

    public init(vocabURL: URL, maxLength: Int = 512) throws {
        let text = try String(contentsOf: vocabURL, encoding: .utf8)
        var v: [String: Int] = [:]
        var i = 0
        for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
            // vocab.txt is one token per line, index == line number.
            if i == 0 && line.isEmpty { continue }
            v[String(line)] = i
            i += 1
        }
        vocab = v
        unkId = v["[UNK]"] ?? 100
        clsId = v["[CLS]"] ?? 101
        sepId = v["[SEP]"] ?? 102
        self.maxLength = maxLength
    }

    // MARK: - Character classification (HF parity)

    private static func isControl(_ c: Character) -> Bool {
        if c == "\t" || c == "\n" || c == "\r" { return false }
        guard let s = c.unicodeScalars.first else { return false }
        switch s.properties.generalCategory {
        case .control, .format: return true
        default: return false
        }
    }

    private static func isWhitespace(_ c: Character) -> Bool {
        if c == "\t" || c == "\n" || c == "\r" { return true }
        return c.isWhitespace
    }

    /// HF `_is_punctuation`: the ASCII punctuation ranges plus Unicode P*.
    /// Deliberately excludes symbols (S* categories).
    private static func isPunctuation(_ c: Character) -> Bool {
        if let v = c.unicodeScalars.first?.value {
            switch v {
            case 33...47, 58...64, 91...96, 123...126: return true
            default: break
            }
        }
        return c.isPunctuation
    }

    private static func isCJK(_ v: UInt32) -> Bool {
        (0x4E00...0x9FFF).contains(v) || (0x3400...0x4DBF).contains(v)
            || (0x20000...0x2A6DF).contains(v) || (0x2A700...0x2B73F).contains(v)
            || (0x2B740...0x2B81F).contains(v) || (0x2B820...0x2CEAF).contains(v)
            || (0xF900...0xFAFF).contains(v) || (0x2F800...0x2FA1F).contains(v)
    }

    // MARK: - Normalization (clean + CJK space + lowercase + strip accents)

    private func normalize(_ chars: [Character]) -> [NormChar] {
        var out: [NormChar] = []
        out.reserveCapacity(chars.count + 8)
        for (i, c) in chars.enumerated() {
            guard let scalar = c.unicodeScalars.first else { continue }
            let v = scalar.value
            if v == 0 || v == 0xFFFD || Self.isControl(c) { continue }
            if Self.isWhitespace(c) {
                out.append(NormChar(ch: " ", origin: i))
                continue
            }
            if Self.isCJK(v) {
                out.append(NormChar(ch: " ", origin: i))
                out.append(NormChar(ch: c, origin: i))
                out.append(NormChar(ch: " ", origin: i))
                continue
            }
            // lowercase, then strip accents (drop Mn combining marks)
            let lowered = String(c).lowercased().decomposedStringWithCanonicalMapping
            for s in lowered.unicodeScalars where s.properties.generalCategory != .nonspacingMark {
                out.append(NormChar(ch: Character(s), origin: i))
            }
        }
        return out
    }

    // MARK: - Encode

    /// Encode into ids + offsets with [CLS] ... [SEP], truncated to `maxLength`.
    public func encode(_ text: String) -> [Token] {
        let chars = Array(text)
        let norm = normalize(chars)

        // Split into words on whitespace, with punctuation as its own word.
        var words: [[NormChar]] = []
        var current: [NormChar] = []
        func endWord() {
            if !current.isEmpty { words.append(current); current = [] }
        }
        for nc in norm {
            if nc.ch == " " {
                endWord()
            } else if Self.isPunctuation(nc.ch) {
                endWord()
                words.append([nc])
            } else {
                current.append(nc)
            }
        }
        endWord()

        var pieces: [Token] = [Token(id: clsId, range: nil)]
        let bodyLimit = maxLength - 1  // reserve room for [SEP]

        for word in words {
            let wordRange = word.first!.origin..<(word.last!.origin + 1)
            if word.count > maxInputCharsPerWord {
                pieces.append(Token(id: unkId, range: wordRange))
                if pieces.count >= bodyLimit { break }
                continue
            }

            // greedy longest-match wordpiece, carrying offsets
            var s = 0
            var sub: [Token] = []
            var bad = false
            while s < word.count {
                var e = word.count
                var matchId: Int?
                var matchEnd = e
                while s < e {
                    var cand = String(word[s..<e].map { $0.ch })
                    if s > 0 { cand = "##" + cand }
                    if let id = vocab[cand] { matchId = id; matchEnd = e; break }
                    e -= 1
                }
                guard let id = matchId else { bad = true; break }
                let lo = word[s].origin
                let hi = word[matchEnd - 1].origin + 1
                sub.append(Token(id: id, range: lo..<hi))
                s = matchEnd
            }

            if bad {
                pieces.append(Token(id: unkId, range: wordRange))
            } else {
                pieces.append(contentsOf: sub)
            }
            if pieces.count >= bodyLimit { break }
        }

        pieces.append(Token(id: sepId, range: nil))
        return pieces
    }
}
