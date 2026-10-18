import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

/// Allosaurus on 48 GB Macs (2026-10-15): residency and the user advisory must be sized against the
/// SYSTEM WIRE BUDGET (`vm.global_user_wire_limit` − already wired), not only RAM / Metal working set.
/// Measured: ~33.5 GiB of wire headroom → 22-28 tok/s with kernel wire rejections; ~37.5 GiB → 46-48 tok/s.
final class WireBudgetPolicyTests: XCTestCase {
    private let gib = 1024 * 1024 * 1024
    private let allosaurusBytes = 36_972_157_624  // gpuResidentWeightBytes from the bundle headers

    func testNoWireHeadroomReadingKeepsExistingPolicy() {
        let without = modelResidencyWiredTarget(
            modelBytes: allosaurusBytes, workingSet: 115_448_725_504, physicalMemory: 128 * gib)
        let withNil = modelResidencyWiredTarget(
            modelBytes: allosaurusBytes, workingSet: 115_448_725_504, physicalMemory: 128 * gib,
            wireHeadroom: nil)
        XCTAssertEqual(without, withNil)
        XCTAssertNotNil(without)
    }

    func testFortyEightGigabyteMacRequestsTheWholeWorkingSet() {
        // Measured under a 48 GB-like wire budget + RAM pressure: no residency 0.2 tok/s (740 GB page-ins),
        // a set capped to the wire headroom 18 tok/s, a set covering the working set 36-41 tok/s.
        let target = modelResidencyWiredTarget(
            modelBytes: allosaurusBytes, workingSet: 40 * gib, physicalMemory: 48 * gib,
            wireHeadroom: Int(31.3 * Double(gib)))
        XCTAssertNotNil(target, "pinning nothing is the 0.2 tok/s field failure")
        if let target {
            XCTAssertGreaterThanOrEqual(
                target, allosaurusBytes + 5 * gib / 2, "weights + KV/verify scratch must be in the set")
            XCTAssertLessThanOrEqual(target, 40 * gib, "never above Metal's working set")
        }
    }

    func testFortyEightGigabyteMacStillPinsWhatMetalAllowsWhenTheModelIsLarger() {
        let target = modelResidencyWiredTarget(
            modelBytes: allosaurusBytes, workingSet: 33 * gib, physicalMemory: 48 * gib)
        XCTAssertEqual(target, 33 * gib, "a partial set (18 tok/s) beats none (0.2 tok/s)")
    }

    func testWireHeadroomDoesNotShrinkTheResidencyRequest() {
        let unbounded = modelResidencyWiredTarget(
            modelBytes: allosaurusBytes, workingSet: 115_448_725_504, physicalMemory: 128 * gib)
        let tight = modelResidencyWiredTarget(
            modelBytes: allosaurusBytes, workingSet: 115_448_725_504, physicalMemory: 128 * gib,
            wireHeadroom: 30 * gib)
        XCTAssertEqual(unbounded, tight)
    }

    func testXNUDefaultWireLimitMatchesMeasuredDevices() {
        XCTAssertEqual(xnuDefaultUserWireLimit(physicalMemory: 51_539_607_552), 40_716_289_966)  // 48 GiB M4 Pro
        XCTAssertEqual(xnuDefaultUserWireLimit(physicalMemory: 137_438_953_472), 116_823_110_451)  // 128 GiB M5 Max
    }

    func testReporterFortyEightGigabyteMacExactValues() {
        // tpae, M4 Pro 48 GB: hw.memsize, recommendedMaxWorkingSetSize, vm.global_user_wire_limit.
        let target = modelResidencyWiredTarget(
            modelBytes: allosaurusBytes, workingSet: 40_200_896_512, physicalMemory: 51_539_607_552,
            wireHeadroom: 40_716_289_966 - 8_140_000_000, systemWireLimit: 40_716_289_966)
        XCTAssertEqual(target, 40_200_896_512, "the whole working set, bounded by Metal's ceiling — never nil")
    }

    func testKernelCapBoundsExistingTargetWhenMetalCeilingIsHigher() {
        let defaultCap = xnuDefaultUserWireLimit(physicalMemory: 48 * gib)
        for cap in [defaultCap, 30 * gib, 0] {
            let target = modelResidencyWiredTarget(
                modelBytes: 29 * gib, workingSet: 40 * gib, physicalMemory: 48 * gib,
                systemWireLimit: cap)
            XCTAssertEqual(target, cap > 0 ? cap : nil)
        }
    }

    func testSmallMacKeepsTheFreezeGuard() {
        // osaurus#2612: 16 GB Macs keep the conservative reserve.
        XCTAssertNil(modelResidencyWiredTarget(
            modelBytes: 9 * gib, workingSet: Int(10.6 * Double(gib)), physicalMemory: 16 * gib))
    }

    func testHeadroomIsLimitMinusWiredAndNeverNegative() {
        XCTAssertEqual(
            SystemWireBudget(userWireLimitBytes: 100, gpuWiredLimitBytes: nil, wiredBytes: 30,
                             wireLimitRejections: 0).headroomBytes, 70)
        XCTAssertEqual(
            SystemWireBudget(userWireLimitBytes: 10, gpuWiredLimitBytes: nil, wiredBytes: 30,
                             wireLimitRejections: 0).headroomBytes, 0)
        XCTAssertNil(
            SystemWireBudget(userWireLimitBytes: nil, gpuWiredLimitBytes: nil, wiredBytes: 30,
                             wireLimitRejections: nil).headroomBytes,
            "an unreadable limit is unknown, never a limit")
    }

    func testAdvisoryNamesShortfallOnlyWhenTheWorkingSetDoesNotFit() {
        let tight = SystemWireBudget(
            userWireLimitBytes: 40_716_289_966, gpuWiredLimitBytes: nil,
            wiredBytes: Int(7.06 * Double(gib)), wireLimitRejections: 0)
        let report = ModelWireBudgetReport(
            modelBytes: allosaurusBytes, budget: tight, residencyTargetBytes: nil)
        XCTAssertTrue(report.exceedsWireBudget)
        XCTAssertNotNil(report.advisory)
        XCTAssertTrue(report.advisory?.contains("The model still loads") == true, "advisory, never a refusal")

        let roomy = SystemWireBudget(
            userWireLimitBytes: 108 * gib, gpuWiredLimitBytes: nil, wiredBytes: 6 * gib,
            wireLimitRejections: 0)
        let fits = ModelWireBudgetReport(
            modelBytes: allosaurusBytes, budget: roomy, residencyTargetBytes: 50 * gib)
        XCTAssertFalse(fits.exceedsWireBudget)
        XCTAssertNil(fits.advisory)

        let unknown = SystemWireBudget(
            userWireLimitBytes: nil, gpuWiredLimitBytes: nil, wiredBytes: nil, wireLimitRejections: nil)
        XCTAssertNil(ModelWireBudgetReport(
            modelBytes: allosaurusBytes, budget: unknown, residencyTargetBytes: nil).advisory)
    }

    /// Allosaurus 0.2 on 48 GB Macs (2026-10-18): the MLX freed-buffer pool is wired too. Sized from RAM it was
    /// ~6 GB, and weights + KV + scratch + pool overran the wire budget (decode 46-64 -> 23-34 tok/s). The ceiling
    /// keeps the pool inside what the generation leaves; it never applies on 64 GiB+ Macs or an unknown budget.
    func testAllocatorCacheCeilingFitsTheWireBudgetLeftByTheGeneration() {
        let v02Bytes = 28_838_118_008  // Allosaurus 0.2 candidate v3 GPU-resident weights (n-gram table on SSD)
        let mac48 = SystemWireBudget(
            userWireLimitBytes: 40_716_289_966, gpuWiredLimitBytes: nil,
            wiredBytes: Int(7.06 * Double(gib)), wireLimitRejections: 0)
        let target = modelResidencyWiredTarget(
            modelBytes: v02Bytes, workingSet: 40_200_896_512, physicalMemory: 48 * gib,
            wireHeadroom: mac48.headroomBytes, systemWireLimit: mac48.userWireLimitBytes)
        let report = ModelWireBudgetReport(
            modelBytes: v02Bytes, budget: mac48, residencyTargetBytes: target, physicalMemoryBytes: 48 * gib)
        let ceiling = try! XCTUnwrap(report.allocatorCacheCeilingBytes)
        let wireable = min(mac48.headroomBytes!, target!)
        XCTAssertLessThanOrEqual(report.estimatedGenerationBytes + ceiling, wireable)
        XCTAssertGreaterThanOrEqual(ceiling, ModelWireBudgetReport.minimumAllocatorCacheCeilingBytes)
        XCTAssertLessThan(ceiling, 2 * gib, "the RAM-sized 6 GB pool is what overran the budget")

        // A model that leaves no room still gets the minimum pool (a 0-byte pool re-allocates every step).
        let tight = ModelWireBudgetReport(
            modelBytes: allosaurusBytes, budget: mac48, residencyTargetBytes: nil, physicalMemoryBytes: 48 * gib)
        XCTAssertEqual(tight.allocatorCacheCeilingBytes, ModelWireBudgetReport.minimumAllocatorCacheCeilingBytes)

        let mac128 = SystemWireBudget(
            userWireLimitBytes: 116_823_110_451, gpuWiredLimitBytes: nil, wiredBytes: 7 * gib,
            wireLimitRejections: 0)
        XCTAssertNil(ModelWireBudgetReport(
            modelBytes: v02Bytes, budget: mac128, residencyTargetBytes: 46_017_987_192,
            physicalMemoryBytes: 128 * gib).allocatorCacheCeilingBytes, "no change on 64 GiB+ Macs")
        let unknown = SystemWireBudget(
            userWireLimitBytes: nil, gpuWiredLimitBytes: nil, wiredBytes: nil, wireLimitRejections: nil)
        XCTAssertNil(ModelWireBudgetReport(
            modelBytes: v02Bytes, budget: unknown, residencyTargetBytes: nil,
            physicalMemoryBytes: 48 * gib).allocatorCacheCeilingBytes, "an unknown budget is never a limit")
    }

    func testLiveSnapshotReadsTheBudgetWithoutPrivileges() {
        let budget = SystemWireBudget.snapshot()
        #if os(macOS)
        XCTAssertNotNil(budget.userWireLimitBytes)
        XCTAssertNotNil(budget.wiredBytes)
        XCTAssertNotNil(budget.wireLimitRejections)
        if let limit = budget.userWireLimitBytes {
            XCTAssertGreaterThan(limit, 0)
            XCTAssertLessThanOrEqual(UInt64(limit), ProcessInfo.processInfo.physicalMemory)
        }
        #endif
    }
}
