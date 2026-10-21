// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import XCTest

@testable import MLXLMCommon

final class DFlash2DrafterResolverTests: XCTestCase {
    private func model() throws -> DFlash2DraftModel {
        DFlash2DraftModel(
            try DFlash2Configuration(json: [
                "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 4, "intermediate_size": 16,
                "vocab_size": 256, "num_target_layers": 1, "is_causal": false,
                "layer_types": ["full_attention"],
                "rope_parameters": ["rope_theta": 10000, "rope_type": "default"],
                "dflash_config": [
                    "block_size": 4, "conv_kernel_size": 2,
                    "conv_group_size": 2, "mask_token_id": 250, "selector_rank": 4,
                    "selector_top_k": 2, "target_layer_ids": [0],
                ],
            ]))
    }

    func testSamePathSharesOneLoadWhileOtherPathsRemainIndependent() throws {
        let shared = try model()
        let firstEntered = DispatchSemaphore(value: 0)
        let duplicateEntered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        let lock = NSLock()
        var count = 0
        let resolver = DFlash2DrafterResolver { path in
            if path.lastPathComponent == "a" {
                lock.lock()
                count += 1
                let current = count
                lock.unlock()
                if current == 1 {
                    firstEntered.signal()
                    _ = release.wait(timeout: .now() + 5)
                } else {
                    duplicateEntered.signal()
                }
            }
            return shared
        }
        let a = URL(fileURLWithPath: "/tmp/dflash-resolver-a/a")
        let finished = expectation(description: "both callers complete")
        finished.expectedFulfillmentCount = 2
        DispatchQueue.global().async {
            XCTAssertTrue((try? resolver.drafter(at: a)) === shared)
            finished.fulfill()
        }
        XCTAssertEqual(firstEntered.wait(timeout: .now() + 5), .success)
        DispatchQueue.global().async {
            XCTAssertTrue((try? resolver.drafter(at: a)) === shared)
            finished.fulfill()
        }
        XCTAssertEqual(duplicateEntered.wait(timeout: .now() + 0.1), .timedOut)
        // Must complete while a is deliberately blocked in its loader.
        XCTAssertTrue(
            try resolver.drafter(at: URL(fileURLWithPath: "/tmp/dflash-resolver-b")) === shared)
        release.signal()
        wait(for: [finished], timeout: 5)
        XCTAssertEqual(count, 1)
    }

    func testEvictionDuringLoadDoesNotRepublishOrReplaceNewerLoad() throws {
        let old = try model()
        let new = try model()
        for all in [false, true] {
            let entered = DispatchSemaphore(value: 0)
            let release = DispatchSemaphore(value: 0)
            let lock = NSLock()
            var count = 0
            let resolver = DFlash2DrafterResolver { _ in
                lock.lock()
                count += 1
                let current = count
                lock.unlock()
                if current == 1 {
                    entered.signal()
                    _ = release.wait(timeout: .now() + 5)
                    return old
                }
                return new
            }
            let path = URL(fileURLWithPath: "/tmp/dflash-resolver-evict")
            let finished = expectation(description: "old request keeps its model")
            DispatchQueue.global().async {
                XCTAssertTrue((try? resolver.drafter(at: path)) === old)
                finished.fulfill()
            }
            XCTAssertEqual(entered.wait(timeout: .now() + 5), .success)
            if all { resolver.evictAll() } else { resolver.evict(path: path) }
            XCTAssertTrue(try resolver.drafter(at: path) === new)
            release.signal()
            wait(for: [finished], timeout: 5)
            XCTAssertTrue(try resolver.drafter(at: path) === new)
            XCTAssertEqual(count, 2)
        }
    }

    func testFailedLoadCanBeRetried() throws {
        enum Failed: Error { case expected }
        let loaded = try model()
        var count = 0
        let resolver = DFlash2DrafterResolver { _ in
            count += 1
            if count == 1 { throw Failed.expected }
            return loaded
        }
        let path = URL(fileURLWithPath: "/tmp/dflash-resolver-retry")
        XCTAssertThrowsError(try resolver.drafter(at: path))
        XCTAssertTrue(try resolver.drafter(at: path) === loaded)
        XCTAssertTrue(try resolver.drafter(at: path) === loaded)
        XCTAssertEqual(count, 2)
    }
}
