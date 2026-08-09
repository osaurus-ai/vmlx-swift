// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import XCTest

@testable import MLX

class CompiledFunctionLifetimeTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    /// The backend compiler cache entry for a compiled function retains the
    /// full traced graph — including every constant captured at trace time
    /// (for model regions: the weights). That entry is only erased in
    /// `CompiledFunction.deinit`, so the object MUST deallocate once callers
    /// drop it. A self-capture inside the cached trampoline closures
    /// (`cachedInnerClosure`/`cachedCompiledClosure`) forms a retain cycle
    /// that leaks the entry — and with it the whole model — on unload.
    func testCompiledFunctionDeallocatesAfterCall() {
        weak var weakState: CompiledFunction?
        autoreleasepool {
            let state = CompiledFunction(
                inputs: [], outputs: [], shapeless: false, trusted: true
            ) {
                [$0[0] * 2]
            }
            weakState = state
            XCTAssertNotNil(weakState)

            // Populate the cached backend closures (the state-free fast path):
            // this is the exact path that formed the cycle.
            let out = state.call([MLXArray(3)])
            XCTAssertEqual(out[0].item(Int.self), 6)
            let out2 = state.call([MLXArray(5)])
            XCTAssertEqual(out2[0].item(Int.self), 10)
        }
        XCTAssertNil(
            weakState,
            "CompiledFunction must deallocate after release — its deinit is the "
                + "only place the backend compile-cache entry (traced constants, "
                + "i.e. model weights) is freed")
    }
}
