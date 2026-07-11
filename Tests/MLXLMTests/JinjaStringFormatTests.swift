// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import VMLXJinja
import Testing

/// The vendored Jinja engine's Python `str.format` subset — added for the
/// official Hunyuan v3 chat template, which builds every special-token
/// string with `'<x{}>'.format(HYTK)`. All 8 template-smoke scenarios failed
/// with `Cannot call non-function value` before this existed.
struct JinjaStringFormatTests {

    private func render(_ template: String, _ context: [String: Value] = [:]) throws -> String {
        try Template(template).render(context)
    }

    @Test("auto-numbered {} fields (the Hunyuan template shape)")
    func autoNumbered() throws {
        let out = try render(
            "{% set HYTK = ':opensource' %}{{ '<think{}>'.format(HYTK) }}")
        #expect(out == "<think:opensource>")
    }

    @Test("multiple auto fields consume positional args in order")
    func multipleAuto() throws {
        let out = try render("{{ '{}-{}'.format('a', 'b') }}")
        #expect(out == "a-b")
    }

    @Test("indexed and named fields")
    func indexedAndNamed() throws {
        #expect(try render("{{ '{1}{0}'.format('a', 'b') }}") == "ba")
        #expect(try render("{{ '{x}!'.format(x='hey') }}") == "hey!")
    }

    @Test("literal brace escapes")
    func braceEscapes() throws {
        #expect(try render("{{ '{{{}}}'.format('v') }}") == "{v}")
    }

    @Test("integer arguments render in canonical form")
    func intArgs() throws {
        #expect(try render("{{ 'n={}'.format(3) }}") == "n=3")
    }

    @Test("format specs are rejected, not mis-rendered")
    func specsRejected() {
        #expect(throws: (any Error).self) {
            _ = try render("{{ '{:>8}'.format('x') }}")
        }
    }

    @Test("running out of positional arguments raises")
    func missingArgRaises() {
        #expect(throws: (any Error).self) {
            _ = try render("{{ '{} {}'.format('only-one') }}")
        }
    }
}
