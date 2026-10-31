import Foundation
import Testing
import VMLXJinja

@Suite("Foundation values preserve JSON scalar types")
struct JinjaValueBridgeTests {
    @Test("booleans remain distinct from zero and one after Foundation decoding")
    func booleanAndNumberTypes() throws {
        let data = Data(#"{"flag_false":false,"flag_true":true,"zero":0,"one":1,"rate":0.5}"#.utf8)
        let value = try Value(any: JSONSerialization.jsonObject(with: data))
        let source = "{{ data.flag_false | tojson }}|{{ data.flag_true | tojson }}|{{ data.zero | tojson }}|{{ data.one | tojson }}|{{ data.rate | tojson }}"
        #expect(try Template(source).render(["data": value]) == "false|true|0|1|0.5")
    }

    @Test("boxed integral-valued doubles remain floating point")
    func boxedDoubleType() throws {
        let value = try Value(any: NSNumber(value: 1.0))
        #expect(try Template("{{ value | tojson }}").render(["value": value]) == "1.0")
    }
}
