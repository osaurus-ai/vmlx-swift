// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Opt-in loader-component measurement. No model weights or token generation.
@Suite("Bonsai2 packed expansion profile", .serialized)
struct JangTernaryPackedProfileTests {
    @Test(
        "bounded dense shapes expand exact words with measured wall time",
        .enabled(if: ProcessInfo.processInfo.environment["BONSAI2_PACKED_PROFILE"] == "1"))
    func measuredExpansion() throws {
        try MLXMetalTestLock.withLock {
            for (rows, width) in [(4096, 5120), (2048, 17408)] {
                let groups = width / 128
                var bytes = Array(repeating: UInt8(121), count: rows * groups * 26)
                for index in stride(from: 25, to: bytes.count, by: 26) { bytes[index] = 13 }
                let packed = MLXArray(bytes, [rows, groups * 26])
                let scales = MLXArray.ones([rows, groups], dtype: .float16)
                MLX.eval(packed, scales)
                for iteration in 0 ..< 5 {
                    let start = Date.timeIntervalSinceReferenceDate
                    let result = try expandJangTernaryPacked(packed, scales: scales)
                    let milliseconds = (Date.timeIntervalSinceReferenceDate - start) * 1000
                    // Outside the timed region: validate every output word.
                    #expect(result.weight.dtype == .uint32)
                    #expect(result.weight.shape == [rows, width / 16])
                    #expect(MLX.all(result.weight .== UInt32(0x5555_5555)).item(Bool.self))
                    #expect(result.scales === scales)
                    #expect(MLX.all(result.biases .== Float(-1)).item(Bool.self))
                    let record: [String: Any] = [
                        "kind": "bonsai2_packed_expansion", "rows": rows, "width": width,
                        "codes": rows * width, "iteration": iteration,
                        "milliseconds": milliseconds,
                        "device": String(describing: Device.defaultDevice()),
                    ]
                    let data = try JSONSerialization.data(
                        withJSONObject: record, options: [.sortedKeys])
                    print(String(decoding: data, as: UTF8.self))
                }
            }
        }
    }
}
