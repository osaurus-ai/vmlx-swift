// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
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

    func testInsufficientWindowDoesNotMasqueradeAsACompleteHit() {
        let store = DFlash2ContextStore()
        let prompt = Array(0 ..< 100)
        store.store(tokens: prompt, salt: nil, rows: rows(40 ..< 100))
        XCTAssertNil(store.rows(endingAt: 70, of: prompt, salt: nil, minimumRows: 40))
        XCTAssertEqual(
            positions(store.rows(endingAt: 70, of: prompt, salt: nil, minimumRows: 30)),
            Array(40 ..< 70))
    }

    func testSearchContinuesPastAnIncompleteNewerEntry() {
        let store = DFlash2ContextStore()
        let prompt = Array(0 ..< 100)
        store.store(tokens: Array(prompt.prefix(70)), salt: nil, rows: rows(0 ..< 70))
        store.store(tokens: prompt, salt: nil, rows: rows(40 ..< 100))
        XCTAssertEqual(
            positions(store.rows(endingAt: 70, of: prompt, salt: nil, minimumRows: 40)),
            Array(0 ..< 70))
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

    func testAuxiliaryAndApproximateRowsDoNotEvictConversation() {
        let store = DFlash2ContextStore()
        let prompt = Array(0 ..< 20)
        store.store(tokens: prompt, salt: nil, rows: rows(0 ..< 20))
        for i in 0 ..< 12 {
            store.store(tokens: [100 + i, 1, 2], salt: nil, rows: rows(0 ..< 3), intent: .auxiliary)
            store.store(tokens: [200 + i, 1, 2], salt: nil, rows: rows(0 ..< 3), isExact: false)
        }
        XCTAssertEqual(positions(store.rows(endingAt: 20, of: prompt, salt: nil)), prompt)
        XCTAssertNil(store.rows(endingAt: 3, of: [100, 1, 2], salt: nil))
    }

    func testDiskFeatureContractRejectsWrongLayersBoundaryWidthAndCoverage() throws {
        let store = DFlash2ContextStore()
        let prompt = Array(0 ..< 100)
        store.store(tokens: prompt, salt: nil, rows: rows(40 ..< 100))
        let payload = try XCTUnwrap(store.diskPayload(endingAt: 100, of: prompt, salt: nil, layers: [9]))
        func read(_ boundary: Int = 100, _ end: Int = 99, _ count: Int = 59, _ layers: [Int] = [9], _ width: Int = 1) -> MLXArray? {
            DFlash2ContextStore.diskRows(payload, cacheBoundary: boundary, endingAt: end,
                                        minimumRows: count, layers: layers, width: width)
        }
        XCTAssertEqual(positions(read()), Array(40 ..< 99))
        XCTAssertNil(read(99))
        XCTAssertNil(read(100, 99, 60))
        XCTAssertNil(read(100, 99, 59, [17]))
        XCTAssertNil(read(100, 99, 59, [9], 2))
        var malformed = payload
        malformed["dflash2_context_meta"] = MLXArray([Int32(2), 100, 60, 1])
        XCTAssertNil(DFlash2ContextStore.validatedDiskPayload(malformed, boundary: 100))
    }

    func testFeaturesSurviveDiskReopenInsideTheTargetEntry() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("dflash-context-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let config = CacheCoordinatorConfig(usePagedCache: false, enableDiskCache: true,
                                           pagedBlockSize: 4, diskCacheMaxGB: 0.01,
                                           diskCacheDir: root, modelKey: "dflash-context-test")
        let prompt = Array(0 ..< 20)
        let store = DFlash2ContextStore()
        store.store(tokens: prompt, salt: "image-a", rows: rows(0 ..< 20))
        let kv = KVCacheSimple()
        _ = kv.update(keys: MLXArray.ones([1, 1, 20, 4]), values: MLXArray.ones([1, 1, 20, 4]))
        let writer = CacheCoordinator(config: config)
        writer.storeAfterGeneration(promptTokens: prompt, perLayerData: [], ssmStates: nil,
                                    cache: [kv], mediaSalt: "image-a",
                                    dflashContext: store.diskPayload(endingAt: 20, of: prompt, salt: "image-a", layers: [9]))
        store.removeAll()
        let reader = CacheCoordinator(config: config)
        guard case .hit(let matched, _, let detail, _, _, let disk) = reader.fetch(tokens: prompt + [20], mediaSalt: "image-a") else {
            return XCTFail("Expected the persisted target entry")
        }
        XCTAssertEqual(matched, 20)
        XCTAssertEqual(detail, .disk)
        let arrays = try XCTUnwrap(disk)
        XCTAssertEqual(positions(DFlash2ContextStore.diskRows(arrays, cacheBoundary: 20,
                         endingAt: 20, minimumRows: 20, layers: [9], width: 1)), prompt)
        var target: [any KVCache] = [KVCacheSimple()]
        XCTAssertEqual(restoreFromDiskArrays(arrays, into: &target), 20)
        XCTAssertEqual(target[0].offset, 20)
        XCTAssertFalse(reader.hasDurableDiskEntry(tokens: prompt, mediaSalt: "image-b"))
        // Same target boundary and tensor dimensions, different tap layer:
        // must replace the validated entry, not skip the store by layout.
        store.store(tokens: prompt, salt: "image-a", rows: rows(0 ..< 20))
        reader.storeAfterGeneration(promptTokens: prompt, perLayerData: [], ssmStates: nil,
                                    cache: [kv], mediaSalt: "image-a",
                                    dflashContext: store.diskPayload(endingAt: 20, of: prompt, salt: "image-a", layers: [17]))
        let reopened = CacheCoordinator(config: config)
        guard case .hit(_, _, _, _, _, let replaced) = reopened.fetch(tokens: prompt + [20], mediaSalt: "image-a") else {
            return XCTFail("Expected replacement entry")
        }
        let updated = try XCTUnwrap(replaced)
        XCTAssertNotNil(DFlash2ContextStore.diskRows(updated, cacheBoundary: 20, endingAt: 20,
                                                    minimumRows: 20, layers: [17], width: 1))
        XCTAssertNil(DFlash2ContextStore.diskRows(updated, cacheBoundary: 20, endingAt: 20,
                                                 minimumRows: 20, layers: [9], width: 1))
    }
}
