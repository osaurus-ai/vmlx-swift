import Foundation
import Testing

/// Pins the two properties of `AddedTokenTrie` that a plausible "simplification" would break, and
/// that no other test in-tree would catch.
///
/// Source-level because `AddedTokenTrie` and `String.split(by:)` are both internal to `Tokenizers`,
/// which has no test target in this tree. The behavioural differential that found the real bug was
/// run out-of-tree (PR #246); this keeps the fix from silently reverting.
@Suite("AddedTokenTrie source coverage")
struct AddedTokenTrieSourceCoverageTests {
    private static func source(_ relativePath: String) throws -> String {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(contentsOf: repoRoot.appending(path: relativePath), encoding: .utf8)
    }

    /// The walk MUST be over Unicode scalars, not `Character`s.
    ///
    /// `Array(text)` yields extended grapheme clusters, so a token's final scalar fuses with a
    /// following combining mark and the token stops matching entirely — `"<|im_end|>\u{0301}rest"`
    /// was left unsplit while the regex arm split it correctly. Scalars are also the level the
    /// regex operates at, which is what makes the two arms identical by construction.
    @Test("trie walks unicode scalars, not grapheme clusters")
    func trieWalksUnicodeScalars() throws {
        let source = try Self.source("Vendors/swift-transformers/Sources/Tokenizers/AddedTokenTrie.swift")
        #expect(source.contains("[Unicode.Scalar: Node]"))
        #expect(source.contains("text.unicodeScalars"))
        #expect(source.contains("for scalar in token.unicodeScalars"))
        // The exact regression: walking `Character`s.
        #expect(!source.contains("let characters = Array(text)"))
        #expect(!source.contains("[Character: Node]"))
    }

    /// Longest match must win, which requires CONTINUING the walk past a terminal node rather than
    /// returning at the first one. The regex gets this from longest-first alternation ordering; if
    /// the trie returned early, `</think>` would split as `<` + `/think>` on any token set where a
    /// shorter token is a prefix of a longer one.
    @Test("trie keeps walking past a terminal node so the longest token wins")
    func triePrefersLongestMatch() throws {
        let source = try Self.source("Vendors/swift-transformers/Sources/Tokenizers/AddedTokenTrie.swift")
        #expect(source.contains("if node.isTerminal { longest = index }"))
    }
}
