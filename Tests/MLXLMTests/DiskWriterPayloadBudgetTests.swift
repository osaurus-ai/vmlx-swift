import Foundation
import MLX
import XCTest
@testable import MLXLMCommon

final class DiskWriterPayloadBudgetTests: XCTestCase {
    func testIncomingPayloadCannotOvershootSharedBudget() throws {
        let budget = DiskWriterPayloadBudget()
        var first: DiskWriterPayloadBudget.Reservation? = try XCTUnwrap(budget.reserve(bytes: 8, limit: 12))
        XCTAssertNil(budget.reserve(bytes: 8, limit: 12))
        var second = budget.reserve(bytes: 4, limit: 12)
        XCTAssertNotNil(second)
        XCTAssertEqual(budget.currentBytes, 12)
        XCTAssertNil(budget.reserve(bytes: 1, limit: 12))
        first = nil
        XCTAssertEqual(budget.currentBytes, 4)
        XCTAssertNotNil(budget.reserve(bytes: 8, limit: 12))
        second = nil
        XCTAssertEqual(budget.currentBytes, 0)
    }

    func testOversizeAndInvalidRequestsNeverConsumeBudget() {
        let budget = DiskWriterPayloadBudget()
        XCTAssertNil(budget.reserve(bytes: 13, limit: 12))
        XCTAssertNil(budget.reserve(bytes: -1, limit: 12))
        XCTAssertNil(budget.reserve(bytes: 1, limit: -1))
        XCTAssertEqual(budget.currentBytes, 0)
    }

    func testReservationFollowsPayloadOwnerLifetime() throws {
        let budget = DiskWriterPayloadBudget()
        var producer: DiskWriterPayloadBudget.Reservation? = try XCTUnwrap(budget.reserve(bytes: 8, limit: 12))
        var writer = producer
        producer = nil
        XCTAssertEqual(budget.currentBytes, 8)
        XCTAssertNil(budget.reserve(bytes: 8, limit: 12))
        writer = nil
        XCTAssertEqual(budget.currentBytes, 0)
    }
    func testTwoCacheInstancesShareOneIncomingPayloadBudget() throws {
        let lock = lockSerializedMLXTest()
        defer { lock.unlock() }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        DiskCache.inflightByteBudgetOverrideForTesting = 12 * 1024
        defer { DiskCache.inflightByteBudgetOverrideForTesting = nil }
        let first = DiskCache(cacheDir: root.appendingPathComponent("first"), maxSizeBytes: 1_000_000)
        let second = DiskCache(cacheDir: root.appendingPathComponent("second"), maxSizeBytes: 1_000_000)
        first.writeBehind = true; second.writeBehind = true
        first.writeDelayForTesting = 0.5
        let arrays = ["keys": MLXArray(Array(repeating: Int32(7), count: 2048))]
        first.store(tokens: [1, 2, 3], arrays: arrays)
        XCTAssertEqual(first.pendingWriteCount, 1)
        XCTAssertEqual(DiskWriterPayloadBudget.shared.currentBytes, 8192)
        second.store(tokens: [4, 5, 6], arrays: arrays)
        XCTAssertEqual(second.writeBehindStores, 0)
        XCTAssertEqual(second.writeBehindBudgetFallbacks, 1)
        XCTAssertTrue(second.hasDurableEntry(tokens: [4, 5, 6]))
        XCTAssertTrue(first.waitForPendingPayloadRelease())
        XCTAssertTrue(first.hasDurableEntry(tokens: [1, 2, 3]))
        XCTAssertEqual(DiskWriterPayloadBudget.shared.currentBytes, 0)
    }

}
