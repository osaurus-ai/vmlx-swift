import Testing
import VMLXJinja

@Suite("Native Spark template primitives")
struct JinjaSparkTemplateTests {
    @Test func commentWhitespaceControls() throws {
        #expect(try Template("a \n{#- comment -#}\n b").render([:]) == "ab")
        #expect(try Template("a \n{# comment #}\n b").render([:]) == "a \n\n b")
        #expect(try Template("a \n{#- comment #}\n b").render([:]) == "a\n b")
        #expect(try Template("a \n{# comment -#}\n b").render([:]) == "a \nb")
    }

    @Test func quotedCommentLookingTextIsData() throws {
        #expect(try Template("{{ 'a {#- literal -#} b' }}").render([:]) == "a {#- literal -#} b")
    }

    @Test func neighborsAreValuesWithUndefinedEdges() throws {
        let source = "{% for x in xs %}{{ loop.previtem|default('L') }}:{{ x }}:{{ loop.nextitem|default('R') }};{% endfor %}"
        #expect(try Template(source).render(["xs": .array([.int(1), .int(2), .int(3)])]) == "L:1:2;1:2:3;2:3:R;")
        #expect(try Template(source).render(["xs": .array([.int(7)])]) == "L:7:R;")
        #expect(try Template(source).render(["xs": .array([])]) == "")
    }

    @Test func nativeToolGroupingUsesNeighborRoles() throws {
        let source = "{% for m in messages %}{% if m.role == 'tool' %}{% if loop.previtem is undefined or loop.previtem.role != 'tool' %}<T>{% endif %}{{ m.content }}{% if loop.nextitem is undefined or loop.nextitem.role != 'tool' %}</T>{% endif %}{% endif %}{% endfor %}"
        let messages: [Value] = [.object(["role": .string("user")]),
            .object(["role": .string("tool"), "content": .string("a")]),
            .object(["role": .string("tool"), "content": .string("b")]),
            .object(["role": .string("user")]),
            .object(["role": .string("tool"), "content": .string("c")])]
        #expect(try Template(source).render(["messages": .array(messages)]) == "<T>ab</T><T>c</T>")
    }

    @Test func filteredArrayNeighborsAndElseUseIncludedItems() throws {
        let source = "{% for x in xs if x > 1 %}{{ loop.index }}/{{ loop.length }}:{{ loop.previtem|default('L') }}:{{ x }}:{{ loop.nextitem|default('R') }};{% else %}empty{% endfor %}"
        #expect(try Template(source).render(["xs": .array([.int(1), .int(2), .int(0), .int(3)])]) == "1/2:L:2:3;2/2:2:3:R;")
        #expect(try Template(source).render(["xs": .array([.int(0), .int(1)])]) == "empty")
    }

    @Test func nestedLoopsRestoreTheOuterNeighborScope() throws {
        let source = "{% for x in xs %}{% for y in ys %}{{ loop.previtem|default('L') }}{% endfor %}:{{ loop.nextitem|default('R') }};{% endfor %}"
        #expect(try Template(source).render(["xs": .array([.int(1), .int(2)]), "ys": .array([.int(3)])]) == "L:2;L:R;")
    }
}
