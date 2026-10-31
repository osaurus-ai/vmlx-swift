import Testing
import VMLXJinja

@Suite("Jinja namespace members")
struct JinjaNamespaceMemberTests {
    @Test("namespace items is stored data rather than a dictionary method")
    func itemsAttributeDoesNotResolveToMethod() throws {
        let source =
            "{% set carry = namespace(items=[]) %}{% set result = [{'type':'text','text':'7680'}] %}{% if carry.items %}{{ carry.items + result }}{% else %}{{ result[0].text }}{% endif %}"
        #expect(try Template(source).render([:]) == "7680")
    }

    @Test("namespace attributes persist across loops and method-name collisions")
    func accumulatesAcrossLoop() throws {
        let source =
            "{% set ns = namespace(items=[], keys='key', values='value', get='get') %}{% for value in ['a','b'] %}{% set ns.items = ns.items + [value] %}{% endfor %}{{ ns.items | join(',') }}|{{ ns.keys }}|{{ ns.values }}|{{ ns.get }}"
        #expect(try Template(source).render([:]) == "a,b|key|value|get")
    }

    @Test("ordinary mapping items remains callable")
    func dictionaryMethodsRemainMethods() throws {
        #expect(
            try Template("{% for key, value in data.items() %}{{ key }}={{ value }}{% endfor %}")
                .render(["data": .object(["answer": .int(7680)])]) == "answer=7680")
    }

    @Test("empty namespace is truthy and not a mapping")
    func namespaceHasItsOwnType() throws {
        #expect(
            try Template(
                "{% set ns = namespace() %}{{ ns is mapping }}|{{ ns is iterable }}|{% if ns %}yes{% else %}no{% endif %}"
            )
            .render([:]) == "false|false|yes")
    }
}
