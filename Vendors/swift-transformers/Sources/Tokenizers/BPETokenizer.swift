//
//  BPETokenizer.swift
//  CoreMLBert
//
//  Created by Julien Chaumond on 18/07/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import Foundation
import VMLXHub

/// A pair of byte/token strings used in Byte-Pair Encoding (BPE) merge operations.
///
/// Compared and hashed literally, by UTF-8 bytes, as Hugging Face tokenizers does. Swift's
/// `String ==` is canonical instead, so canonically equivalent merges (577 groups in Gemma's table)
/// collapsed onto one key carrying the rank of the group's last member.
struct BytePair: Hashable, Sendable {
    let a: String
    let b: String

    /// Stores both halves as native UTF-8, which the byte comparisons read fastest. Merge-table
    /// strings arrive bridged from `NSString`, mostly as UTF-16; a native string costs a flag test.
    init(_ a: String, _ b: String) {
        var a = a
        var b = b
        a.makeContiguousUTF8()
        b.makeContiguousUTF8()
        self.a = a
        self.b = b
    }

    init(tuple: [String]) {
        self.init(tuple[0], tuple[1])
    }

    static func == (lhs: BytePair, rhs: BytePair) -> Bool {
        lhs.a.utf8.elementsEqual(rhs.a.utf8) && lhs.b.utf8.elementsEqual(rhs.b.utf8)
    }

    func hash(into hasher: inout Hasher) {
        Self.hashBytes(of: a, into: &hasher)
        Self.hashBytes(of: b, into: &hasher)
    }

    /// The string's UTF-8 bytes, after their count so that ("ab", "c") and ("a", "bc") differ.
    private static func hashBytes(of string: String, into hasher: inout Hasher) {
        var string = string
        string.withUTF8 { bytes in
            hasher.combine(bytes.count)
            hasher.combine(bytes: UnsafeRawBufferPointer(bytes))
        }
    }
}

/// A Byte-Pair Encoding (BPE) tokenizer implementation.
///
/// BPE tokenizers learn to merge the most frequently occurring pairs of characters
/// or character sequences. This implementation supports various BPE-based models
/// including GPT-2, RoBERTa, and other transformer models.
class BPETokenizer: PreTrainedTokenizerModel, @unchecked Sendable {
    let bpeRanks: [BytePair: Int]
    private let tokensToIds: [NSString: Int]
    private let idsToTokens: [Int: NSString]

    /// The total number of tokens in the vocabulary.
    var vocabCount: Int { tokensToIds.count }

    /// The beginning-of-sequence token string, if defined.
    let bosToken: String?

    /// The numeric ID of the beginning-of-sequence token, if defined.
    let bosTokenId: Int?

    /// The end-of-sequence token string, if defined.
    let eosToken: String?

    /// The numeric ID of the end-of-sequence token, if defined.
    let eosTokenId: Int?

    /// The unknown token string used for out-of-vocabulary words.
    let unknownToken: String?

    /// The numeric ID of the unknown token.
    let unknownTokenId: Int?

    /// Whether consecutive unknown tokens should be fused together.
    let fuseUnknownTokens: Bool

    static func mergesFromConfig(_ config: Config?) -> [[String]]? {
        guard let config else { return nil }

        if let merges = config.array() {
            return merges.reduce(into: [[String]]()) { result, element in
                if let val: [String] = element.get() { // New format (pushed with tokenizers >= 0.20.0): each merge is a list of 2 items
                    result.append(val)
                }
                if let val: String = element.get() { // legacy
                    result.append(val.unicodeScalars.split(separator: " ", omittingEmptySubsequences: false).map { String($0) })
                }
            }
        }

        return nil
    }

    /// Initializes a BPE tokenizer from configuration data.
    ///
    /// - Parameters:
    ///   - tokenizerConfig: The tokenizer configuration
    ///   - tokenizerData: The tokenizer data containing vocabulary and merges
    ///   - addedTokens: Additional tokens to include in the vocabulary
    /// - Throws: `TokenizerError` if required configuration is missing
    required init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int]) throws {
        guard let merges = Self.mergesFromConfig(tokenizerData.model.merges) else { fatalError("BPETokenizer requires merges") }
        guard let vocab = tokenizerData.model.vocab.dictionary() else {
            throw TokenizerError.missingVocab
        }
        var bpeRanks: [BytePair: Int] = [:]
        for (i, merge) in merges.enumerated() {
            let bp = BytePair(tuple: merge)
            bpeRanks[bp] = i
        }
        self.bpeRanks = bpeRanks

        let addedTokens = addedTokens.reduce(into: [BinaryDistinctString: Config]()) { result, element in
            result[BinaryDistinctString(element.key)] = .init(element.value)
        }
        tokensToIds = vocab.merging(addedTokens) { $1 }.reduce(into: [NSString: Int]()) { result, element in
            result[element.key.nsString] = element.value.integer()
        }

        idsToTokens = tokensToIds.reduce(into: [Int: NSString]()) { result, element in
            result[element.value] = element.key
        }

        // Populate tokens
        if let unknownToken = TokenizerModel.unknownToken(from: tokenizerConfig) {
            self.unknownToken = unknownToken
            unknownTokenId = tokensToIds[unknownToken as NSString]
        } else {
            unknownToken = nil
            unknownTokenId = nil
        }

        eosToken = addedTokenAsString(tokenizerConfig.eosToken)
        eosTokenId = eosToken == nil ? nil : tokensToIds[eosToken! as NSString]

        bosToken = addedTokenAsString(tokenizerConfig.bosToken)
        bosTokenId = bosToken == nil ? nil : tokensToIds[bosToken! as NSString]

        fuseUnknownTokens = tokenizerConfig.fuseUnk.boolean(or: false)
    }

    /// Converts a token string to its corresponding numeric ID.
    ///
    /// - Parameter token: The token string to convert
    /// - Returns: The numeric ID, or the unknown token ID if not found
    func convertTokenToId(_ token: String) -> Int? {
        tokensToIds[token as NSString] ?? unknownTokenId
    }

    /// Converts a numeric token ID back to its string representation.
    ///
    /// - Parameter id: The numeric token ID to convert
    /// - Returns: The token string, or nil if the ID is invalid
    func convertIdToToken(_ id: Int) -> String? {
        idsToTokens[id] as String?
    }

    func byteEncode(text: String) -> [String] {
        let RE = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        let tokens = text.ranges(of: RE).map { String(text[$0]) }
        return tokens.map { token -> String in
            return Array(token.utf8).compactMap { byteEncoder[$0] }.joined()
        }
    }

    func hexaEncode(text: String) -> [String] {
        let RE = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        let tokens = text.ranges(of: RE).map { String(text[$0]) }
        return tokens.flatMap { token -> [String] in
            return Array(token.utf8).map { String(format: "<0x%02X>", $0) }
        }
    }

    private func getPairs(word: [String]) -> Set<BytePair> {
        var s = Set<BytePair>()
        for i in 0..<word.count - 1 {
            let bp = BytePair(
                word[i],
                word[i + 1]
            )
            s.insert(bp)
        }
        return s
    }

    /// Byte-Pair Encoding of a single pre-token.
    ///
    /// Linear-ish merge: a doubly-linked list of symbols plus a min-heap of
    /// candidate adjacent merges keyed by `(rank, leftIndex)`. Each merge is
    /// O(log n) and there are O(n) merges, so an N-symbol token is O(n log n)
    /// rather than O(n²) — this matters for long whitespace-free pre-tokens
    /// (e.g. compact tool JSON), which a naive per-round rescan tokenizes in
    /// seconds.
    ///
    /// Merges happen in rank-rounds: within a round every non-overlapping
    /// occurrence of the current min-rank pair is merged (left to right)
    /// before any pair created by those merges is admitted, as in the original
    /// per-round rescan, which the differential test pins. Hugging Face
    /// tokenizers admits a new pair at once instead, so the two differ where a
    /// merged pair outranks its own components, as in SentencePiece whitespace
    /// runs (rank(\t\t,\t) < rank(\t,\t)): on Gemma's table, 32 tabs give
    /// 16 + 16 here and 31 + 1 there. Because the heap is keyed
    /// `(rank, leftIndex)`, same-rank occurrences pop in left-to-right order
    /// for free. Stale heap entries (a node consumed by an earlier merge, or
    /// whose pair rank no longer matches the candidate) are skipped on pop.
    ///
    /// Returns the pieces in order, and `[]` for an empty token. They stay an
    /// array, as in upstream swift-transformers #355, because joining them on
    /// " " and splitting again loses boundaries: a piece that starts with a
    /// combining mark or an emoji modifier forms one grapheme cluster with the
    /// space before it, and a piece containing U+0020 is split at it, or lost
    /// if it holds nothing else.
    func bpe(token: String) -> [String] {
        // Seed from Unicode scalars, as Hugging Face tokenizers does. `Array(token)` yields grapheme
        // clusters, and no merge can then produce a token that starts or ends inside one. Upstream
        // swift-transformers #355 made the same fix; AddedTokenTrie.split(_:) notes the same trap.
        let parts0 = token.unicodeScalars.map { String($0) }
        let n = parts0.count
        if n <= 1 { return parts0 }

        var parts = parts0
        var prev = [Int](repeating: 0, count: n)
        var next = [Int](repeating: 0, count: n)
        var alive = [Bool](repeating: true, count: n)
        for i in 0..<n {
            prev[i] = i - 1
            next[i] = (i + 1 == n) ? -1 : (i + 1)
        }

        // Binary min-heap of (rank, left), ordered by rank then left index.
        var heap: [(rank: Int, left: Int)] = []
        heap.reserveCapacity(n)
        func before(_ a: (rank: Int, left: Int), _ b: (rank: Int, left: Int)) -> Bool {
            a.rank != b.rank ? a.rank < b.rank : a.left < b.left
        }
        func push(_ left: Int) {
            guard left >= 0 else { return }
            let r = next[left]
            guard r >= 0, let rank = bpeRanks[BytePair(parts[left], parts[r])] else { return }
            heap.append((rank, left))
            var i = heap.count - 1
            while i > 0 {
                let parent = (i - 1) / 2
                if before(heap[i], heap[parent]) {
                    heap.swapAt(i, parent)
                    i = parent
                } else {
                    break
                }
            }
        }
        func pop() -> (rank: Int, left: Int)? {
            guard let top = heap.first else { return nil }
            let last = heap.removeLast()
            if !heap.isEmpty {
                heap[0] = last
                var i = 0
                let count = heap.count
                while true {
                    let l = 2 * i + 1
                    let r = 2 * i + 2
                    var m = i
                    if l < count, before(heap[l], heap[m]) { m = l }
                    if r < count, before(heap[r], heap[m]) { m = r }
                    if m == i { break }
                    heap.swapAt(i, m)
                    i = m
                }
            }
            return top
        }

        for i in 0..<n where next[i] >= 0 { push(i) }

        // Adjacencies created during a round, pushed only once the round ends so
        // a new lower-rank pair cannot preempt the round's remaining merges.
        var pending: [Int] = []
        while let head = pop() {
            let roundRank = head.rank
            var cand: (rank: Int, left: Int)? = head
            pending.removeAll(keepingCapacity: true)
            repeat {
                let l = cand!.left
                if alive[l] {
                    let r = next[l]
                    // Skip stale entries: the live pair at this node must still
                    // carry the round's rank.
                    if r >= 0, alive[r],
                        let curRank = bpeRanks[BytePair(parts[l], parts[r])], curRank == roundRank
                    {
                        // Merge r into l; r leaves the list.
                        parts[l] = parts[l] + parts[r]
                        alive[r] = false
                        let rn = next[r]
                        next[l] = rn
                        if rn >= 0 { prev[rn] = l }

                        // Defer adjacencies created by this merge to the next round.
                        pending.append(prev[l])
                        pending.append(l)
                    }
                }
                // Drain remaining same-rank occurrences (they sit at the heap top).
                if let top = heap.first, top.rank == roundRank {
                    cand = pop()
                } else {
                    cand = nil
                }
            } while cand != nil

            for left in pending { push(left) }
        }

        // Walk the surviving list from the head (node 0 is never consumed —
        // it is never the right element of any merge).
        var result: [String] = []
        result.reserveCapacity(n)
        var i = 0
        while i >= 0 {
            result.append(parts[i])
            i = next[i]
        }
        return result
    }

    /// Tokenizes input text using the BPE algorithm.
    ///
    /// - Parameter text: The input text to tokenize
    /// - Returns: An array of BPE token strings
    func tokenize(text: String) -> [String] {
        var tokens: [String] = []
        for token in bpe(token: text) {
            if convertTokenToId(token) != unknownTokenId {
                tokens.append(token)
            } else {
                // TODO: if config.byte_fallback is False, append the unknown token instead
                tokens.append(contentsOf: hexaEncode(text: token))
            }
        }
        return tokens
    }
}
