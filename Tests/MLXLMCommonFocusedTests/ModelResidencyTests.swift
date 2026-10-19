import Foundation
import XCTest

@testable import MLXLMCommon

final class ModelResidencyTests: XCTestCase {
    private final class Backend: @unchecked Sendable {
        var limits: [Int] = []
        var succeed = true
        func apply(_ limit: Int) -> Bool {
            limits.append(limit)
            return succeed
        }
    }

    private func request(_ gib: Int, enabled: Bool = true, ram: Int = 128) -> ModelResidency.Request {
        .init(weightBytes: gib << 30, enabled: enabled, physicalMemory: ram << 30,
              workingSet: (ram * 9 / 10) << 30, systemWireLimit: (ram * 85 / 100) << 30)
    }

    func testSecondModelAndUnloadRestoreSurvivorWithoutZeroingIt() {
        let backend = Backend()
        let registry = ModelResidency(apply: backend.apply)
        var a: ModelResidency.Lease? = registry.acquire(request(28)).lease
        let original = backend.limits.last!
        var b: ModelResidency.Lease? = registry.acquire(request(2)).lease
        XCTAssertGreaterThan(backend.limits.last!, original)
        XCTAssertFalse(backend.limits.contains(0))
        withExtendedLifetime(b) {}
        b = nil
        XCTAssertEqual(backend.limits.last, original)
        withExtendedLifetime(a) {}
        a = nil
        XCTAssertEqual(backend.limits.last, 0)
    }

    func testOffSecondModelDoesNotUnpinLiveEnabledModel() {
        let backend = Backend()
        let registry = ModelResidency(apply: backend.apply)
        var a: ModelResidency.Lease? = registry.acquire(request(28)).lease
        let original = backend.limits.last!
        let b = registry.acquire(request(2, enabled: false)).lease
        XCTAssertGreaterThanOrEqual(backend.limits.last!, original)
        withExtendedLifetime(a) {}
        a = nil
        XCTAssertEqual(backend.limits.last, 0)
        withExtendedLifetime(b) {}
    }

    func testDuplicateContextsShareLeaseAndIndependentLoadsCountTwice() {
        let backend = Backend()
        let registry = ModelResidency(apply: backend.apply)
        var a: ModelResidency.Lease? = registry.acquire(request(10)).lease
        var copy = a
        let original = backend.limits.last!
        a = nil
        XCTAssertEqual(backend.limits.last, original)
        var independent: ModelResidency.Lease? = registry.acquire(request(10)).lease
        XCTAssertEqual(backend.limits.last, 36 << 30)
        withExtendedLifetime(independent) {}
        independent = nil
        XCTAssertEqual(backend.limits.last, original)
        withExtendedLifetime(copy) {}
        copy = nil
        XCTAssertEqual(backend.limits.last, 0)
    }

    func testFailedLoadScopeReleasesReservation() {
        enum Failure: Error { case load }
        let backend = Backend()
        let registry = ModelResidency(apply: backend.apply)
        let a = registry.acquire(request(28)).lease
        let original = backend.limits.last!
        func fail() throws {
            let pending = registry.acquire(request(2)).lease
            defer { withExtendedLifetime(pending) {} }
            throw Failure.load
        }
        XCTAssertThrowsError(try fail())
        XCTAssertEqual(backend.limits.last, original)
        withExtendedLifetime(a) {}
    }

    func testNonFittingSecondModelDoesNotClearSmallMacSurvivor() {
        let a = request(2, ram: 32)
        let b = request(30, ram: 32)
        XCTAssertEqual(ModelResidency.target(for: [a, b]), ModelResidency.target(for: [a]))
        XCTAssertGreaterThan(ModelResidency.target(for: [a]), 0)
        XCTAssertEqual(ModelResidency.target(for: [request(28, enabled: false)]), 0)
    }

    func testFailedBackendApplicationIsRetriedOnNextTransition() {
        let backend = Backend()
        backend.succeed = false
        let registry = ModelResidency(apply: backend.apply)
        let a = registry.acquire(request(28))
        XCTAssertFalse(a.applied)
        backend.succeed = true
        let b = registry.acquire(request(2))
        XCTAssertTrue(b.applied)
        XCTAssertEqual(backend.limits.count, 2)
        withExtendedLifetime((a, b)) {}
    }
}
