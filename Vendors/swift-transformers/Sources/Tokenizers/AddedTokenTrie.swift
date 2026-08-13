import Foundation

/// Splits text on a set of literal added-token strings in time independent of how many there are.
///
/// The regex path this replaces builds ONE `NSRegularExpression` whose pattern is every added token
/// joined by `|`, each in its own capture group, and runs it on every `tokenize(text:)` call. That is
/// O(text x tokenCount) twice over: ICU has no literal-alternation optimisation, and
/// `String.split(by:)` then walks `(0..<match.numberOfRanges).reversed()` per match doing an
/// `NSRange` -> `Range<String.Index>` conversion at each step.
///
/// With a handful of added tokens that is invisible. Discrete-codebook multimodal models change the
/// scale: Apertus 1.5 Omni declares 136,366 added tokens (131,072 `<|visual token N|>` plus 4,096
/// `<|audio token N|>`), and each special token in a prompt then costs ~28 SECONDS — a 76-token
/// prompt took 218 s to reach first token while decode was a healthy 66 tok/s.
///
/// A trie makes the scan O(text) regardless of token count, because shared prefixes (`<|visual token `)
/// collapse into one path.
///
/// **Semantics are matched exactly**, so callers see no behavioural change:
///   * longest match wins at each position — the regex sorts alternatives longest-first and ICU
///     alternation is first-match-wins, which is the same rule;
///   * matched separators are emitted as their own sections, interleaved with the text between them;
///   * empty sections are omitted.
///
/// **Guarded**: only usable when no added token sets `lstrip`/`rstrip`. Those add `\s*` around the
/// token in the regex, which is not a literal match, so the regex path stays for those bundles.
final class AddedTokenTrie {

    private final class Node {
        var children: [Unicode.Scalar: Node] = [:]
        var isTerminal = false
    }

    private let root = Node()

    init(tokens: [String]) {
        for token in tokens where !token.isEmpty {
            var node = root
            for scalar in token.unicodeScalars {
                if let next = node.children[scalar] {
                    node = next
                } else {
                    let next = Node()
                    node.children[scalar] = next
                    node = next
                }
            }
            node.isTerminal = true
        }
    }

    /// End index of the LONGEST token matching at `start`, or nil when none does.
    private func longestMatch(_ scalars: [Unicode.Scalar], from start: Int) -> Int? {
        var node = root
        var index = start
        var longest: Int?
        while index < scalars.count, let next = node.children[scalars[index]] {
            node = next
            index += 1
            if node.isTerminal { longest = index }   // keep going: a longer token may still match
        }
        return longest
    }

    /// Text split into alternating non-token and token sections, empties omitted.
    ///
    /// Walks UNICODE SCALARS, not `Character`s. `Array(text)` yields extended grapheme clusters, so
    /// a token's final scalar can fuse with a following combining mark into one cluster and the walk
    /// then cannot match the token at all: `"<|im_end|>\u{0301}rest"` left the token unrecognised
    /// while the regex arm correctly split it out. Scalars are also the level the regex operates at,
    /// so this keeps the two arms identical by construction rather than by coincidence.
    /// (Found by a 16-case differential against the regex arm; this was the one divergence.)
    func split(_ text: String) -> [String] {
        let scalars = Array(text.unicodeScalars)
        var sections: [String] = []
        var pending = String.UnicodeScalarView()
        var index = 0
        while index < scalars.count {
            if let end = longestMatch(scalars, from: index) {
                if !pending.isEmpty {
                    sections.append(String(pending))
                    pending = String.UnicodeScalarView()
                }
                sections.append(String(String.UnicodeScalarView(scalars[index ..< end])))
                index = end
            } else {
                pending.append(scalars[index])
                index += 1
            }
        }
        if !pending.isEmpty { sections.append(String(pending)) }
        return sections.isEmpty ? [text] : sections
    }
}
