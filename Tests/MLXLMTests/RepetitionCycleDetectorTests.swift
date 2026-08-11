import Foundation
import Testing

@testable import MLXLMCommon

/// The guard has to fire on the observed collapse and stay silent on every
/// ordinary shape that happens to repeat. False positives here truncate real
/// answers, so the negative cases matter more than the positive one.
@Suite("Degenerate repetition detector")
struct RepetitionCycleDetectorTests {

    /// Verbatim from the live Raptor turn that spent its whole token budget.
    static let observedUnit =
        "The answer is AppleScript; it begins with `use AppleScript version`. "
        + "I do not generate or repeat the request. "

    @Test("fires on the observed collapse")
    func detectsObservedLoop() throws {
        let text = String(repeating: Self.observedUnit, count: 6)
        let cycle = try #require(RepetitionCycleDetector.cycle(in: text))
        #expect(cycle.unit == Self.observedUnit)
        #expect(cycle.repeats >= RepetitionCycleDetector.minimumRepeats)
    }

    @Test("reports the shortest primitive unit, not a multiple of it")
    func reportsShortestUnit() throws {
        // Primitive: no shorter string repeats to build it.
        let unit = "the same sentence again and again, ok? "  // 39 chars
        let cycle = try #require(RepetitionCycleDetector.cycle(in: String(repeating: unit, count: 8)))
        #expect(cycle.unit == unit, "reported \(cycle.unit.count)-char unit: \(cycle.unit.debugDescription)")
    }

    /// A unit that is itself built from a shorter period is filler, however
    /// long it looks — `"abcdefgh" x 4` is a 32-character unit with period 8.
    @Test("a non-primitive unit is treated as filler")
    func nonPrimitiveUnitIsFiller() {
        let unit = String(repeating: "abcdefgh", count: 4)
        #expect(RepetitionCycleDetector.cycle(in: String(repeating: unit, count: 8)) == nil)
        #expect(RepetitionCycleDetector.minimalPeriod(of: Array(unit)) == 8)
    }

    @Test("streams to the same verdict as the whole-string form")
    func streamingMatchesWholeString() throws {
        var detector = RepetitionCycleDetector()
        var fired: RepetitionCycleDetector.Cycle?
        for _ in 0 ..< 8 {
            // Arrive in small pieces, the way detokenized chunks do.
            for piece in Self.observedUnit.chunked(into: 7) where fired == nil {
                fired = detector.feed(piece)
            }
        }
        let cycle = try #require(fired)
        #expect(cycle.unit == Self.observedUnit)
    }

    // MARK: - Must NOT fire

    @Test("three repeats are not enough")
    func threeRepeatsAreBelowThreshold() {
        #expect(RepetitionCycleDetector.cycle(in: String(repeating: Self.observedUnit, count: 3)) == nil)
    }

    @Test(
        "short repeating filler never qualifies",
        arguments: [
            String(repeating: "-", count: 400),
            String(repeating: "= ", count: 200),
            String(repeating: "...", count: 140),
            String(repeating: "| | |\n", count: 80),
            String(repeating: "\n", count: 500),
        ])
    func shortFillerIsIgnored(_ text: String) {
        #expect(
            RepetitionCycleDetector.cycle(in: text) == nil,
            "fired on filler: \(text.prefix(12).debugDescription)")
    }

    @Test("ordinary prose does not fire")
    func proseIsIgnored() {
        let prose = """
            A Merkle proof lets a verifier confirm that one leaf belongs to a tree \
            without downloading the whole tree. The prover supplies the audit path: \
            the sibling hash at each level between the leaf and the root. The verifier \
            hashes upward and compares the result against the known root. If they \
            match, the leaf is in the tree; if not, it is not. The cost is logarithmic \
            in the number of leaves rather than linear, which is what makes the scheme \
            practical for large datasets and for clients that cannot store them.
            """
        #expect(RepetitionCycleDetector.cycle(in: prose) == nil)
    }

    /// Repeated *structure* with differing content is the shape most at risk of
    /// a false positive, and it must survive.
    @Test("a table with distinct rows does not fire")
    func tableWithDistinctRowsIsIgnored() {
        var table = "| name | size | kind |\n| --- | --- | --- |\n"
        for i in 0 ..< 40 {
            table += "| entry_\(i) | \(i * 137) bytes | file |\n"
        }
        #expect(RepetitionCycleDetector.cycle(in: table) == nil)
    }

    @Test("repeated code blocks with differing bodies do not fire")
    func codeWithDifferingBodiesIsIgnored() {
        var code = ""
        for i in 0 ..< 30 {
            code += "func step\(i)() -> Int {\n    return \(i) * 3 + 1\n}\n\n"
        }
        #expect(RepetitionCycleDetector.cycle(in: code) == nil)
    }

    @Test("short output is never examined")
    func shortOutputIsIgnored() {
        var detector = RepetitionCycleDetector()
        #expect(detector.feed(String(repeating: "xy", count: 20)) == nil)
    }

    @Test("disabled detector never fires")
    func disabledNeverFires() {
        var detector = RepetitionCycleDetector(isEnabled: false)
        #expect(detector.feed(String(repeating: Self.observedUnit, count: 10)) == nil)
    }

    @Test("VMLX_REPETITION_STOP=0 disables it")
    func environmentOptOut() {
        #expect(!RepetitionCycleDetector.fromEnvironment(["VMLX_REPETITION_STOP": "0"]).isEnabled)
        #expect(RepetitionCycleDetector.fromEnvironment([:]).isEnabled)
    }
}

extension String {
    fileprivate func chunked(into size: Int) -> [String] {
        var out: [String] = []
        var index = startIndex
        while index < endIndex {
            let next = self.index(index, offsetBy: size, limitedBy: endIndex) ?? endIndex
            out.append(String(self[index ..< next]))
            index = next
        }
        return out
    }
}
