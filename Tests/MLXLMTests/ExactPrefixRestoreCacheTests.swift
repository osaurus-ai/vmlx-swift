import Foundation
import MLX
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Forced tool selection skips disk restore only for disk-backed caches WITHOUT exact-prefix restore. Naive-N0.5's
/// model-owned cache is exact (stored K/V/indexer rows only); recurrent, rotating and mixed topologies are not.
final class ExactPrefixRestoreCacheTests: XCTestCase {
    func testWhichTopologiesRestoreExactly() {
        let naive: [KVCache] = [NaiveN05FlashCache(window: nil, requiresIndexer: true),
                                NaiveN05FlashCache(window: 128, requiresIndexer: false)]
        XCTAssertTrue(cacheRequiresDiskBackedCoordinatorRestore(naive), "Naive is disk-backed")
        XCTAssertTrue(cacheRestoreIsExactPrefix(naive), "and restores an exact prefix")

        XCTAssertTrue(cacheRestoreIsExactPrefix([KVCacheSimple(), KVCacheSimple()]))
        XCTAssertFalse(cacheRestoreIsExactPrefix([MambaCache(), KVCacheSimple()]), "recurrent state is path-dependent")
        XCTAssertFalse(cacheRestoreIsExactPrefix([RotatingKVCache(maxSize: 64), KVCacheSimple()]))
        XCTAssertFalse(cacheRestoreIsExactPrefix(naive + [MambaCache()]), "one inexact layer disqualifies the stack")
        XCTAssertTrue(cacheRestoreIsExactPrefix([CacheList(KVCacheSimple(), KVCacheSimple())]))
        XCTAssertFalse(cacheRestoreIsExactPrefix([CacheList(KVCacheSimple(), MambaCache())]))
        XCTAssertFalse(cacheRestoreIsExactPrefix([]))
    }
}
