// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Stage 5 tests — CompilableCacheList.

import Foundation
import MLX
import MLXLMCommon
import XCTest

class CompilableCacheListTests: XCTestCase {

    /// Promoting a CacheList of KVCacheSimple sub-caches produces a
    /// composite where every sub-cache is CompilableKVCache.
    func testPromoteAllSimple() throws {
        let s1 = KVCacheSimple()
        for _ in 0 ..< 5 {
            _ = s1.update(
                keys: MLXArray.ones([1, 2, 1, 16]),
                values: MLXArray.ones([1, 2, 1, 16]))
        }
        let s2 = KVCacheSimple()
        for _ in 0 ..< 3 {
            _ = s2.update(
                keys: MLXArray.ones([1, 2, 1, 16]),
                values: MLXArray.ones([1, 2, 1, 16]))
        }
        let list = CacheList([s1 as KVCache, s2 as KVCache])

        let cmp = CompilableCacheList(from: list)
        XCTAssertEqual(cmp.count, 2)
        XCTAssertTrue(cmp.allSubCachesCompileReady,
            "All KVCacheSimple sub-caches should promote to CompilableKVCache")
        XCTAssertTrue(cmp[0] is CompilableKVCache)
        XCTAssertTrue(cmp[1] is CompilableKVCache)
    }

    /// Promoting a CacheList of RotatingKVCache sub-caches produces
    /// CompilableRotatingKVCache children.
    func testPromoteAllRotating() throws {
        let r1 = RotatingKVCache(maxSize: 32)
        for _ in 0 ..< 5 {
            _ = r1.update(
                keys: MLXArray.ones([1, 2, 1, 16]),
                values: MLXArray.ones([1, 2, 1, 16]))
        }
        let r2 = RotatingKVCache(maxSize: 32)
        for _ in 0 ..< 3 {
            _ = r2.update(
                keys: MLXArray.ones([1, 2, 1, 16]),
                values: MLXArray.ones([1, 2, 1, 16]))
        }
        let list = CacheList([r1 as KVCache, r2 as KVCache])

        let cmp = CompilableCacheList(from: list)
        XCTAssertTrue(cmp.allSubCachesCompileReady)
        XCTAssertTrue(cmp[0] is CompilableRotatingKVCache)
        XCTAssertTrue(cmp[1] is CompilableRotatingKVCache)
    }

    /// Mixed sub-cache types.
    func testPromoteMixedSimpleAndMamba() throws {
        let s = KVCacheSimple()
        for _ in 0 ..< 4 {
            _ = s.update(
                keys: MLXArray.ones([1, 2, 1, 16]),
                values: MLXArray.ones([1, 2, 1, 16]))
        }
        let m = MambaCache()
        m[0] = MLXArray.zeros([1, 4, 3])
        m[1] = MLXArray.zeros([1, 4, 8])

        let list = CacheList([s as KVCache, m as KVCache])

        let cmp = CompilableCacheList(from: list)
        XCTAssertTrue(cmp[0] is CompilableKVCache)
        XCTAssertTrue(cmp[1] is CompilableMambaCache)
        XCTAssertTrue(cmp.allSubCachesCompileReady)
    }

    /// Already-compilable sub-caches pass through unchanged.
    func testAlreadyCompilableSubCachesPassThrough() throws {
        let s = KVCacheSimple()
        for _ in 0 ..< 4 {
            _ = s.update(
                keys: MLXArray.ones([1, 2, 1, 16]),
                values: MLXArray.ones([1, 2, 1, 16]))
        }
        let alreadyCompilable = CompilableKVCache(from: s, maxLength: 256)

        let list = CacheList([alreadyCompilable as KVCache])

        let cmp = CompilableCacheList(from: list)
        XCTAssertTrue(cmp[0] is CompilableKVCache)
        XCTAssertTrue((cmp[0] as AnyObject) === (alreadyCompilable as AnyObject),
            "Already-compilable sub-cache should pass through by identity")
    }

    /// Unknown sub-cache (TQ) passes through + marks not ready.
    func testUnknownSubCachePassesThroughAndMarksNotReady() throws {
        let s = KVCacheSimple()
        for _ in 0 ..< 10 {
            _ = s.update(
                keys: MLXArray.ones([1, 2, 1, 64]),
                values: MLXArray.ones([1, 2, 1, 64]))
        }
        let tq = TurboQuantKVCache.fromSimpleCache(
            s, keyBits: 3, valueBits: 3, residualTokens: 0)

        let list = CacheList([tq as KVCache])

        let cmp = CompilableCacheList(from: list)
        XCTAssertTrue(cmp[0] is TurboQuantKVCache)
        XCTAssertFalse(cmp[0] is CompilableKVCache)
        XCTAssertFalse(cmp.allSubCachesCompileReady,
            "Composite with TQ sub-cache should NOT be compile-ready")
    }

    /// innerState flattens sub-cache inner states in order.
    func testInnerStateFlattens() throws {
        let s1 = KVCacheSimple()
        for _ in 0 ..< 4 {
            _ = s1.update(
                keys: MLXArray.ones([1, 2, 1, 16]),
                values: MLXArray.ones([1, 2, 1, 16]))
        }
        let s2 = KVCacheSimple()
        for _ in 0 ..< 3 {
            _ = s2.update(
                keys: MLXArray.ones([1, 2, 1, 16]),
                values: MLXArray.ones([1, 2, 1, 16]))
        }
        let list = CacheList([s1 as KVCache, s2 as KVCache])
        let cmp = CompilableCacheList(from: list)
        MLX.eval(cmp.state)

        let inner = cmp.innerState()
        // Each CompilableKVCache returns [keys, values, offsetArray] = 3 entries
        XCTAssertEqual(inner.count, 6,
            "Two CompilableKVCache children should produce 6 innerState entries")
    }
}
