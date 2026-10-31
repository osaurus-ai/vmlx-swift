import Foundation
import Testing
import VMLXJinja

@Suite("Upper filter follows the reference value coercion")
struct JinjaUpperReferenceTests {
    @Test("JSON schema values render without a lossy schema rewrite")
    func referenceValues() throws {
        let url = try #require(Bundle.module.url(
            forResource: "upper-reference", withExtension: "json", subdirectory: "Fixtures"))
        let rows = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [[String: Any]])
        #expect(rows.count == 13)
        for row in rows {
            let id = try #require(row["id"] as? String)
            let expected = try #require(row["expected"] as? String)
            let value = try row["context"].map { try Value(any: $0) }
            let source = value == nil ? "{{ missing | upper }}" : "{{ value | upper }}"
            do {
                let result = try Template(source).render(value.map { ["value": $0] } ?? [:])
                #expect(result == expected, "upper reference \(id)")
            } catch {
                Issue.record("Unexpected upper error for \(id): \(error)")
            }
        }
    }
}
