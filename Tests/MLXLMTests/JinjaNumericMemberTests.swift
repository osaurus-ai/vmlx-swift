// Copyright © 2026 osaurus-eval contributors
// SPDX-License-Identifier: MIT
//
// `foo.0` is Jinja's sugar for `foo[0]`, and the parser rejected it.
//
// GLM-5.3's chat template writes `m.content.0.type`. The parser threw "Expected identifier but
// found number", and the CALLER caught that and substituted another family's template — so the
// model was prompted in Gemma-4's format, answered fluently, and never emitted its own turn-end
// token, running to the token limit on every request. A parse error two layers down surfaced as
// "this model rambles".

import Foundation
import VMLXJinja
import Testing

@Suite("Jinja numeric member access")
struct JinjaNumericMemberTests {

    private func render(_ source: String, _ context: [String: Value]) throws -> String {
        try Template(source).render(context)
    }

    @Test("foo.0 indexes a list")
    func numericMemberIndexesAList() throws {
        let out = try render("{{ items.0 }}", ["items": .array([.string("a"), .string("b")])])
        #expect(out == "a")
    }

    @Test("the chain continues after a numeric index")
    func chainAfterNumericIndex() throws {
        // Exactly GLM-5.3's shape: `content.0.type`.
        let out = try render(
            "{{ content.0.type }}",
            ["content": .array([
                .object(["type": .string("tool_reference")]),
                .object(["type": .string("text")]),
            ])])
        #expect(out == "tool_reference")
    }

    @Test("a numeric index means the same as a bracket subscript")
    func matchesBracketForm() throws {
        let context: [String: Value] = [
            "items": .array([.int(10), .int(20), .int(30)])
        ]
        let dotted = try render("{{ items.1 }}", context)
        let bracketed = try render("{{ items[1] }}", context)
        #expect(dotted == bracketed)
        #expect(dotted == "20")
    }

    /// Attribute access must still work — the fix must not swallow ordinary names.
    @Test("named attributes are unaffected")
    func namedAttributesStillWork() throws {
        let out = try render("{{ m.role }}", ["m": .object(["role": .string("user")])])
        #expect(out == "user")
    }
    /// The lexer change must not break ordinary float literals — that is the thing a
    /// "only continue a number before a digit" rule could plausibly have broken.
    @Test("float literals still lex")
    func floatLiteralsStillWork() throws {
        #expect(try render("{{ 1.5 }}", [:]) == "1.5")
        #expect(try render("{{ 0.25 }}", [:]) == "0.25")
        #expect(try render("{{ 2.0 + 0.5 }}", [:]) == "2.5")
        // A number followed by a dot and a NON-digit is an index chain, not a float.
        #expect(
            try render("{{ rows.0.name }}", ["rows": .array([.object(["name": .string("ok")])])])
                == "ok")
    }
}
