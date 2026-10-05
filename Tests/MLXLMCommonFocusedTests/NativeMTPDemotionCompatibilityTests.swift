import XCTest
@testable import MLXLMCommon

final class NativeMTPDemotionCompatibilityTests: XCTestCase {
    func testActualTargetCacheTopologyFailsClosedForHybridBlocks() {
        XCTAssertFalse(NativeMTPTokenIterator.supportsHeadPreservingBlockDemotion([]))
        XCTAssertFalse(NativeMTPTokenIterator.supportsHeadPreservingBlockDemotion([MambaCache()]))
        XCTAssertFalse(NativeMTPTokenIterator.supportsHeadPreservingBlockDemotion([KVCacheSimple(), MambaCache()]))
        XCTAssertTrue(NativeMTPTokenIterator.supportsHeadPreservingBlockDemotion([KVCacheSimple()]))
    }

    func testOnlyConfirmedAlignedTrimmedHistoryCanSurvivePureDemotion() {
        func allowed(aligned: Bool = true, committed: Bool = true,
                     pairs: Bool = true, speculative: Int = 0, repair: Bool = false, compatible: Bool = true) -> Bool {
            NativeMTPTokenIterator.canPreserveHeadOnAdaptiveDemotion(
                alignedAndTrimmable: aligned, targetCommitted: committed, targetCacheCompatible: compatible,
                hasConfirmedPairs: pairs, speculativeRows: speculative, requiresRepair: repair)
        }
        XCTAssertTrue(allowed())
        XCTAssertFalse(allowed(compatible:false), "Hybrid block commit Boolean is insufficient")
        XCTAssertFalse(allowed(aligned:false), "Unaligned/untrimmable/absent head retains reset")
        XCTAssertFalse(allowed(committed:false), "Failed target commit carries no preservation authority")
        XCTAssertFalse(allowed(pairs:false), "Missing confirmed bridge retains reset")
        XCTAssertFalse(allowed(speculative:1), "Untrimmed speculative rows retain reset")
        XCTAssertFalse(allowed(speculative:-1), "Invalid trim bookkeeping fails closed")
        XCTAssertFalse(allowed(repair:true), "Repaired hidden-state path retains reset")
    }
}
