// Offset-aware WordPiece tokenizer for Rampart (BERT uncased).
//
// The vendored `BertTokenizer` only returns token ids; PII redaction needs the
// character span each token covers in the original string, so this carries
// offsets through basic + wordpiece tokenization. Lowercases for vocab lookup
// while keeping original-string indices.
//
// Note: accent stripping is intentionally omitted so character offsets stay
// 1:1 with the input (combining marks would shift indices). PII tokens
// (emails, numbers, IDs, ASCII names) are unaffected.

import Foundation

public struct RampartTokenizer: Sendable {
    public struct Token: Sendable {
        public let id: Int
        /// Character-index range into the original text, or nil for [CLS]/[SEP].
        public let range: Range<Int>?
    }

    private let vocab: [String: Int]
    private let unkId: Int
    private let clsId: Int
    private let sepId: Int
    public let maxLength: Int

    public init(vocabURL: URL, maxLength: Int = 512) throws {
        let text = try String(contentsOf: vocabURL, encoding: .utf8)
        var v: [String: Int] = [:]
        var i = 0
        for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
            let tok = String(line)
            if tok.isEmpty && i == v.count { continue }
            v[tok] = i
            i += 1
        }
        vocab = v
        unkId = v["[UNK]"] ?? 100
        clsId = v["[CLS]"] ?? 101
        sepId = v["[SEP]"] ?? 102
        self.maxLength = maxLength
    }

    private static func isPunctuation(_ c: Character) -> Bool {
        if c.isPunctuation || c.isSymbol { return true }
        guard let v = c.unicodeScalars.first?.value else { return false }
        switch v {
        case 33...47, 58...64, 91...96, 123...126: return true
        default: return false
        }
    }

    /// Encode into ids + offsets with [CLS] ... [SEP], truncated to `maxLength`.
    public func encode(_ text: String) -> [Token] {
        let chars = Array(text)
        var pieces: [Token] = [Token(id: clsId, range: nil)]

        var i = 0
        let bodyLimit = maxLength - 1  // reserve room for [SEP]
        outer: while i < chars.count {
            if chars[i].isWhitespace { i += 1; continue }

            // word boundary: punctuation is its own single-char word
            let start = i
            if Self.isPunctuation(chars[i]) {
                i += 1
            } else {
                while i < chars.count, !chars[i].isWhitespace, !Self.isPunctuation(chars[i]) {
                    i += 1
                }
            }
            let wordChars = Array(chars[start..<i]).map { Character($0.lowercased()) }

            // greedy longest-match wordpiece, carrying offsets
            var s = 0
            var sub: [Token] = []
            var bad = false
            while s < wordChars.count {
                var e = wordChars.count
                var match: String?
                while s < e {
                    var cand = String(wordChars[s..<e])
                    if s > 0 { cand = "##" + cand }
                    if vocab[cand] != nil { match = cand; break }
                    e -= 1
                }
                guard let m = match else { bad = true; break }
                sub.append(Token(id: vocab[m]!, range: (start + s)..<(start + e)))
                s = e
            }
            if bad {
                pieces.append(Token(id: unkId, range: start..<i))
            } else {
                pieces.append(contentsOf: sub)
            }
            if pieces.count >= bodyLimit { break outer }
        }

        pieces.append(Token(id: sepId, range: nil))
        return pieces
    }
}
