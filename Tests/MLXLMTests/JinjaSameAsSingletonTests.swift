import Testing
import VMLXJinja

@Suite("Native Jinja sameas singleton semantics")
struct JinjaSameAsSingletonTests {
    @Test func schemaMappingIsNotABooleanSingleton() throws {
        let nativeExpression = "{% if spec is sameas true %}True{% elif spec is sameas false %}False{% elif spec is mapping %}schema{% else %}other{% endif %}"
        #expect(try Template(nativeExpression).render(["spec": .object(["type": .string("string")])]) == "schema")
        #expect(try Template(nativeExpression).render(["spec": .boolean(true)]) == "True")
        #expect(try Template(nativeExpression).render(["spec": .boolean(false)]) == "False")
        #expect(try Template(nativeExpression).render(["spec": .null]) == "other")
    }

    @Test func booleanAndNoneIdentityRejectOtherTypes() throws {
        let source = "{% for spec in specs %}{% if spec is sameas true %}T{% else %}-{% endif %}{% if spec is sameas false %}F{% else %}-{% endif %}{% if spec is sameas none %}N{% else %}-{% endif %};{% endfor %}"
        let specs: [Value] = [.object(["type": .string("string")]), .boolean(true), .boolean(false),
                              .null, .int(1), .int(0), .double(1), .string("true"), .array([]), .object([:])]
        // Oracle: Python Jinja 3.1.6, whose sameas uses `value is other`.
        #expect(try Template(source).render(["specs": .array(specs)]) == "---;T--;-F-;--N;---;---;---;---;---;---;")
    }

    @Test func negatedSingletonChecksHandleMissingAndContainerValues() throws {
        let source = "{% if spec is not sameas true and spec is not sameas false %}other{% else %}boolean{% endif %}"
        let values: [Value] = [.null, .array([.int(1)]), .object([:]), .int(1), .string("false")]
        for value in values {
            #expect(try Template(source).render(["spec": value]) == "other")
        }
        #expect(try Template(source).render([:]) == "other")
        #expect(try Template(source).render(["spec": .boolean(false)]) == "boolean")
    }

    @Test func numericEqualityDoesNotBecomeCrossTypeIdentity() throws {
        #expect(try Template("{% if 1 is sameas 1.0 %}same{% else %}different{% endif %}").render([:]) == "different")
        #expect(try Template("{% if 1 == 1.0 %}equal{% endif %}").render([:]) == "equal")
    }
}
