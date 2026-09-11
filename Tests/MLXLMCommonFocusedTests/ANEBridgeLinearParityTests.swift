// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// The Neural Engine bridge end to end from Swift: emit an int8 linear with
// ANEMILBuilder, compile + load it through the private framework, eval, and
// compare against a CPU reference. Skips (does not fail) where the private
// framework is absent, so CI without an ANE stays green.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("ANE bridge linear parity")
struct ANEBridgeLinearParityTests {

    @Test("int8 1x1 conv on the ANE matches a CPU reference")
    func int8LinearParity() throws {
        guard ANEProgram.isAvailable else {
            print("ANE unavailable on this host; skipping")
            return
        }
        let k = 1024, n = 1024, rows = 32
        var rng = SystemRandomNumberGenerator()
        let q: [Int8] = (0 ..< n * k).map { _ in Int8(truncatingIfNeeded: Int.random(in: -127 ... 127, using: &rng)) }
        let scale = Float16(1.0 / (127.0 * Float(k).squareRoot()))
        let scales = [Float16](repeating: scale, count: n)
        let x: [Float16] = (0 ..< k * rows).map { _ in Float16(Float.random(in: -1 ... 1, using: &rng)) }

        var b = ANEMILBuilder()
        let x0 = b.input("x0", shape: [1, k, 1, rows])
        let w = b.int8Weight(q, scales: scales, n: n, k: k)
        let y = b.conv(x0, weight: w)
        let mil = b.program(returning: [y])

        let input = try #require(ANEPlane(byteCount: k * rows * 2))
        let output = try #require(ANEPlane(byteCount: n * rows * 2))
        x.withUnsafeBytes { input.pointer.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }

        let program = try ANEProgram(
            name: "linear-parity", mil: mil, weights: b.weights,
            inputs: [input], outputs: [output], cacheDirectory: nil)
        try program.eval()

        // Row 0 (width layout: element [c, r] at c*rows + r).
        var maxAbs: Float = 0, dot: Double = 0, na: Double = 0, nb: Double = 0
        for i in 0 ..< n {
            var acc: Float = 0
            for j in 0 ..< k {
                acc += Float(q[i * k + j]) * Float(scale) * Float(x[j * rows])
            }
            let got = Float(output.fp16[i * rows])
            maxAbs = max(maxAbs, abs(got - acc))
            dot += Double(got * acc); na += Double(got * got); nb += Double(acc * acc)
        }
        let cosine = dot / ((na * nb).squareRoot() + 1e-30)
        print("ANE linear parity: cosine \(cosine) maxAbs \(maxAbs) compile \(program.compileSeconds)s")
        #expect(cosine > 0.9999)
        #expect(maxAbs < 0.01)
    }
}
