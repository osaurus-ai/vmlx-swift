// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import MLX
import MLXRandom
import XCTest

@testable import MLXLMCommon

final class DFlash2ConvolutionKernelTests: XCTestCase {
    func testCausalPaddingAndStridedInputsMatchReference() throws {
        #if canImport(Metal)
            guard Device.defaultDevice() == .gpu else { throw XCTSkip("Metal device required") }
            MLXRandom.seed(829)
            for dtype: DType in [.float32, .float16, .bfloat16] {
                for length in [1, 3, 5, 8, 17] {
                    for taps in [1, 2, 4] {
                        // Transpose forces row-contiguous normalization in
                        // the kernel wrapper rather than assuming raw strides.
                        let hidden = MLXRandom.normal([2, 32, length]).asType(dtype)
                            .transposed(0, 2, 1)
                        let dynamic = MLXRandom.normal([2, length, taps, 4]).asType(dtype)
                        let base = MLXRandom.normal([taps, 32]).asType(dtype)
                        let expected = GroupedDynamicCausalConv.convolveReference(
                            hidden: hidden, dynamic: dynamic, base: base, groupSize: 8)
                        let actual = GroupedDynamicCausalConv.convolveMetal(
                            hidden: hidden, dynamic: dynamic, base: base, groupSize: 8)
                        let error = abs(actual.asType(.float32) - expected.asType(.float32))
                            .max().item(Float.self)
                        XCTAssertLessThanOrEqual(
                            error, dtype == .float32 ? 0.00001 : 0,
                            "dtype=\(dtype) length=\(length) taps=\(taps)")
                    }
                }
            }
        #else
            throw XCTSkip("Metal unavailable")
        #endif
    }
}
