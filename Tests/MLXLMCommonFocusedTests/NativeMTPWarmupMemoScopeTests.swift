// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import XCTest
@testable import MLXLMCommon

final class NativeMTPWarmupMemoScopeTests: XCTestCase {
    func testSuccessfulWarmupBelongsToOneModelInstance() {
        final class Marker {}
        let first = Marker()
        let second = Marker()
        XCTAssertNil(NativeMTPHybridWarmupMemo.verdict(for: first))
        XCTAssertNil(NativeMTPHybridWarmupMemo.verdict(for: second))
        NativeMTPHybridWarmupMemo.record(true, for: first)
        XCTAssertEqual(NativeMTPHybridWarmupMemo.verdict(for: first), true)
        XCTAssertNil(NativeMTPHybridWarmupMemo.verdict(for: second))
    }

    func testMemoDoesNotKeepAnUnloadedModelAlive() {
        final class Marker {}
        weak var unloaded: Marker?
        do {
            let model = Marker()
            unloaded = model
            NativeMTPHybridWarmupMemo.record(true, for: model)
            XCTAssertEqual(NativeMTPHybridWarmupMemo.verdict(for: model), true)
        }
        XCTAssertNil(unloaded)
        XCTAssertNil(NativeMTPHybridWarmupMemo.verdict(for: Marker()))
    }
}
