import XCTest
@testable import MLXLMCommon

/// Exercise the consumed dependency's guard without allocating model/KV data or
/// inducing real host pressure. The kernel level is an independent input.
final class CacheStorePressureTests: XCTestCase {
    private let gib = 1 << 30

    func testWarningAndCriticalVetoDespiteApparentProcessHeadroom() {
        for level in [2, 4] {
            XCTAssertFalse(CacheStoreBudget.canStore(
                cacheBytes: gib,
                activeBytes: 2 * gib,
                budgetBytes: 32 * gib,
                policy: .safeAuto,
                hostPressureLevel: level),
                "Optional copies must stop even when process-only accounting reports headroom")
        }
    }

    func testNormalOrUnavailablePressurePreservesByteBudget() {
        for level: Int? in [1, nil] {
            XCTAssertTrue(CacheStoreBudget.canStore(
                cacheBytes: gib,
                activeBytes: 2 * gib,
                budgetBytes: 32 * gib,
                policy: .safeAuto,
                hostPressureLevel: level))
            XCTAssertFalse(CacheStoreBudget.canStore(
                cacheBytes: gib,
                activeBytes: 31 * gib,
                budgetBytes: 32 * gib,
                policy: .safeAuto,
                hostPressureLevel: level))
        }
    }

    func testWarningStillVetoesWhenPhysicalBudgetIsUnavailable() {
        XCTAssertFalse(CacheStoreBudget.canStore(
            cacheBytes: 1,
            activeBytes: 0,
            budgetBytes: nil,
            hostPressureLevel: 2))
    }

    func testEmptyCacheNeedsNoMaterializationEvenUnderPressure() {
        for level in [2, 4] {
            XCTAssertTrue(CacheStoreBudget.canStore(
                cacheBytes: 0,
                activeBytes: 33 * gib,
                budgetBytes: 32 * gib,
                hostPressureLevel: level))
        }
    }

    func testReclaimedMappedWeightsKeepExistingFootprintAccounting() {
        let active = CacheStoreBudget.activeBytesForStoreBudget(
            physicalFootprintBytes: UInt64(19 * gib),
            mlxActiveBytes: 96 * gib)
        XCTAssertEqual(active, 19 * gib)
        XCTAssertEqual(CacheStoreBudget.activeBytesForStoreBudget(
            physicalFootprintBytes: nil,
            mlxActiveBytes: 37 * gib), 37 * gib)
        XCTAssertTrue(CacheStoreBudget.canStore(
            cacheBytes: gib,
            activeBytes: active,
            budgetBytes: 32 * gib,
            policy: .safeAuto,
            hostPressureLevel: 1))
    }
}
