// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import MLX
import XCTest

@testable import MLXLMCommon

/// Rows are tagged with their absolute position so each lookup can be
/// checked for returning exactly the positions of the restored prefix.
final class DFlash2ContextStoreTests: XCTestCase {
    private func rows(_ positions: Range<Int>) -> MLXArray {
        MLXArray(positions.map(Float.init)).reshaped(1, -1, 1)
    }

    private func positions(_ a: MLXArray?) -> [Int] {
        a.map { $0.reshaped(-1).asArray(Float.self).map { Int($0) } } ?? []
    }

    func testPartialHitReturnsTheRestoredPrefixRows() {
        let store = DFlash2ContextStore()
        let first = Array(0 ..< 100)
        store.store(tokens: first, salt: nil, rows: rows(40 ..< 100))
        let second = first + [7, 7, 7]
        XCTAssertEqual(
            positions(store.rows(endingAt: 100, of: second, salt: nil)), Array(40 ..< 100))
        XCTAssertEqual(positions(store.rows(endingAt: 70, of: second, salt: nil)), Array(40 ..< 70))
        // Before the retained window: nothing to give back.
        XCTAssertNil(store.rows(endingAt: 30, of: second, salt: nil))
    }

    func testFullHitDropsTheRerunLastRow() {
        let store = DFlash2ContextStore()
        let prompt = Array(0 ..< 50)
        store.store(tokens: prompt, salt: nil, rows: rows(0 ..< 50))
        XCTAssertEqual(positions(store.rows(endingAt: 49, of: prompt, salt: nil)), Array(0 ..< 49))
    }

    func testDivergentTokensAndOtherMediaAreNotReused() {
        let store = DFlash2ContextStore()
        store.store(tokens: Array(0 ..< 20), salt: "image-a", rows: rows(0 ..< 20))
        var other = Array(0 ..< 20)
        other[5] = 99
        XCTAssertNil(store.rows(endingAt: 20, of: other, salt: "image-a"))
        XCTAssertNil(store.rows(endingAt: 20, of: Array(0 ..< 20), salt: "image-b"))
        XCTAssertNil(store.rows(endingAt: 20, of: Array(0 ..< 20), salt: nil))
    }

    func testOnlyTheLatestPromptsAreKept() {
        let store = DFlash2ContextStore()
        for i in 0 ... DFlash2ContextStore.capacity {
            store.store(tokens: [1000 + i, 1, 2], salt: nil, rows: rows(0 ..< 3))
        }
        XCTAssertNil(store.rows(endingAt: 3, of: [1000, 1, 2], salt: nil))
        XCTAssertNotNil(
            store.rows(endingAt: 3, of: [1000 + DFlash2ContextStore.capacity, 1, 2], salt: nil))
    }
}
