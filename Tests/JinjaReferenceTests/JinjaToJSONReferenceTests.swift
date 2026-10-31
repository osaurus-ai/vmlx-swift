import Foundation
import Testing
import VMLXJinja

@Suite("HF tojson reference contract")
struct JinjaToJSONReferenceTests {
    private struct Reference: Decodable {
        let id: String
        let template: String
        let expected: String?
        let expected_error: String?
        let string_value: String?
        let float_values: [String]?
    }

    @Test("matches pinned Python/HF JSON rendering and error controls")
    func referenceCorpus() throws {
        let url = try #require(
            Bundle.module.url(
                forResource: "tojson-reference", withExtension: "json", subdirectory: "Fixtures"))
        let rows = try JSONDecoder().decode([Reference].self, from: Data(contentsOf: url))
        #expect(rows.count == 67)
        for row in rows {
            var context: [String: Value] = [:]
            if let text = row.string_value { context["input_text"] = .string(text) }
            if let floats = row.float_values {
                context["input_numbers"] = .array(
                    try floats.map {
                        .double(try #require(Double($0)))
                    })
            }
            if row.expected_error != nil {
                #expect(
                    throws: (any Error).self,
                    "\(row.id) must reject non-JSON input or invalid arguments"
                ) {
                    try Template(row.template).render(context)
                }
            } else {
                do {
                    #expect(
                        try Template(row.template).render(context) == row.expected,
                        "reference case \(row.id)")
                } catch {
                    Issue.record("Unexpected render error in \(row.id): \(error)")
                }
            }
        }
    }

    @Test("unordered host dictionaries retain deterministic cache bytes")
    func unorderedHostDictionaryIsStable() throws {
        let template = try Template("{{ data | tojson }}")
        var a: [String: Any?] = [:]
        a["z"] = 1
        a["a"] = 2
        var b: [String: Any?] = [:]
        b["a"] = 2
        b["z"] = 1
        let first = try template.render(["data": Value(any: a)])
        let second = try template.render(["data": Value(any: b)])
        #expect(first == second)
        #expect(first == #"{"a": 2, "z": 1}"#)
    }
}
