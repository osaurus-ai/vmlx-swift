import Foundation
import XCTest

@testable import MLXLMCommon

final class NativeMTPClockTests: XCTestCase {
    func testUptimeUnitsAndElapsedInterval() {
        XCTAssertEqual(NativeMTPClock.seconds(uptimeNanoseconds: 1_250_000_000), 1.25)
        let first = NativeMTPClock.seconds(uptimeNanoseconds: 10_000_000_000)
        let last = NativeMTPClock.seconds(uptimeNanoseconds: 10_023_300_000)
        XCTAssertEqual(last - first, 0.0233, accuracy: 1e-12)
    }

    func testClockDoesNotMoveBackward() {
        var previous = NativeMTPClock.now()
        for _ in 0..<1000 {
            let current = NativeMTPClock.now()
            XCTAssertGreaterThanOrEqual(current, previous)
            previous = current
        }
    }

    func testIteratorNeverMixesCalendarAndUptimeClocks() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let source = try String(contentsOf: root.appendingPathComponent(
            "Libraries/MLXLMCommon/SpecDec/NativeMTPTokenIterator.swift"), encoding: .utf8)
        XCTAssertFalse(source.contains("Date.timeIntervalSinceReferenceDate"))
        XCTAssertTrue(source.contains("NativeMTPClock.now()"))
    }
}
