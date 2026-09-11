// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Measurement, not a pass/fail gate: how much an ANE eval loop slows a
// decode-shaped MLX 4-bit quantized-matmul loop on the GPU, and vice
// versa. This is the fabric contention an ANE drafter pays while the trunk
// verify streams weights. Prints three arms: GPU alone, ANE alone, both.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("ANE/GPU contention probe")
struct ANEGPUContentionProbeTests {

    @Test("4-bit decode matmul loop vs ANE eval loop")
    func contention() throws {
        guard ANEProgram.isAvailable, ProcessInfo.processInfo.environment["VMLX_ANE_CONTENTION_PROBE"] == "1" else {
            print("skipping (set VMLX_ANE_CONTENTION_PROBE=1 on an ANE host)")
            return
        }
        // GPU arm: 16 layers of 8192x8192 4-bit gs64 weights, M=4 rows (a D3 verify row count).
        let layers = 16, dim = 8192, rows = 4
        let weights: [(MLXArray, MLXArray, MLXArray?)] = (0 ..< layers).map { _ in
            let w = MLXRandom.normal([dim, dim]).asType(.bfloat16)
            let (q, s, b) = MLX.quantized(w, groupSize: 64, bits: 4)
            eval(q, s, b!)
            return (q, s, b)
        }
        var x = MLXRandom.normal([rows, dim]).asType(.bfloat16)
        eval(x)
        func gpuPass() {
            for (q, s, b) in weights {
                x = MLX.quantizedMatmul(x, q, scales: s, biases: b, transpose: true, groupSize: 64, bits: 4)
            }
            eval(x)
        }
        let bytesPerPass = Double(layers) * Double(dim * dim) * 0.5625  // 4 bit + gs64 scales/biases

        // ANE arm: 5120 -> 17408 int8 linear at 32 rows (0.67 ms, ~130 GB/s).
        let k = 5120, n = 17408
        let q8 = [Int8](repeating: 3, count: n * k)
        let scales = [Float16](repeating: Float16(1.0 / 512), count: n)
        var b = ANEMILBuilder()
        let x0 = b.input("x0", shape: [1, k, 1, 32])
        let w = b.int8Weight(q8, scales: scales, n: n, k: k)
        let y = b.conv(x0, weight: w)
        let input = try #require(ANEPlane(byteCount: k * 32 * 2))
        let output = try #require(ANEPlane(byteCount: n * 32 * 2))
        let program = try ANEProgram(name: "contention", mil: b.program(returning: [y]), weights: b.weights,
                                     inputs: [input], outputs: [output], cacheDirectory: nil)
        try program.eval()

        func time(_ body: () -> Void) -> Double {
            let t0 = Date.timeIntervalSinceReferenceDate
            body()
            return Date.timeIntervalSinceReferenceDate - t0
        }
        // warm
        for _ in 0 ..< 5 { gpuPass() }

        let gpuIters = 200, aneIters = 600
        let gpuAlone = time { for _ in 0 ..< gpuIters { gpuPass() } } / Double(gpuIters)
        let aneAlone = time { for _ in 0 ..< aneIters { try? program.eval() } } / Double(aneIters)

        var aneBusy = 0.0, aneCount = 0
        let stop = UnsafeMutablePointer<Bool>.allocate(capacity: 1); stop.pointee = false
        let thread = Thread {
            let t0 = Date.timeIntervalSinceReferenceDate
            while !stop.pointee { try? program.eval(); aneCount += 1 }
            aneBusy = Date.timeIntervalSinceReferenceDate - t0
        }
        thread.start()
        Thread.sleep(forTimeInterval: 0.2)
        let gpuWithANE = time { for _ in 0 ..< gpuIters { gpuPass() } } / Double(gpuIters)
        stop.pointee = true
        while thread.isExecuting { Thread.sleep(forTimeInterval: 0.01) }
        let aneWithGPU = aneBusy / Double(aneCount)

        print(String(format: "GPU 4-bit decode pass: alone %.3f ms (%.0f GB/s) | with ANE %.3f ms (%.0f GB/s) => %.1f%% slower",
                     gpuAlone * 1e3, bytesPerPass / gpuAlone / 1e9, gpuWithANE * 1e3, bytesPerPass / gpuWithANE / 1e9,
                     (gpuWithANE / gpuAlone - 1) * 100))
        print(String(format: "ANE 5120x17408 int8: alone %.3f ms | with GPU %.3f ms => %.1f%% slower",
                     aneAlone * 1e3, aneWithGPU * 1e3, (aneWithGPU / aneAlone - 1) * 100))
        #expect(gpuAlone > 0 && aneAlone > 0)
    }
}
