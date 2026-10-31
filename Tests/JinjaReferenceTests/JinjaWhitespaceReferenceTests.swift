import Testing
import VMLXJinja

@Suite("Explicit whitespace controls follow Jinja delimiters")
struct JinjaWhitespaceReferenceTests {
    @Test("literal braces and structured schema text gain no synthetic space")
    func literalBraces() throws {
        #expect(try Template("{ \n{{- 'x' -}} }").render([:]) == "{x}")
        #expect(try Template("{ \n{%- if true -%}x{%- endif -%} }").render([:]) == "{x}")
        #expect(try Template("{ \n{#- comment -#} }").render([:]) == "{}")
    }

    @Test("quoted delimiter-looking text is never preprocessed")
    func quotedDelimiters() throws {
        #expect(try Template("{{ 'a -}}  b' }}").render([:]) == "a -}}  b")
        #expect(try Template("{{ 'a -%}  b' }}").render([:]) == "a -%}  b")
        #expect(try Template("{{ 'a  {{- b' }}").render([:]) == "a  {{- b")
        #expect(try Template("{{ 'a  {%- b' }}").render([:]) == "a  {%- b")
    }

    @Test("unmarked whitespace and arithmetic remain intact")
    func ordinaryTokens() throws {
        #expect(try Template("{ {{ 'x' }} }").render([:]) == "{ x }")
        #expect(try Template("{{ 8 - 3 }}").render([:]) == "5")
        #expect(try Template("{{ {'x': 1}.x -}} \nend").render([:]) == "1end")
    }
}
