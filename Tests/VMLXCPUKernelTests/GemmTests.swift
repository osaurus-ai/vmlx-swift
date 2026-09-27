// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#if os(Linux)
    import CmlxCPUShim
    import Foundation
    import MLX
    import Testing

    @Suite(.serialized) struct GemmTests {
        /// fp32 GEMMs on both sides of #3019's row split (M >= 16 and M*N*K >= 65536) and of Task 19's
        /// column split below it, which needs at least two 64-column slices (N/64 >= 2), in all four
        /// transpose combinations. Where the column split applies it must happen, and nowhere else.
        @Test(arguments: [1, 3, 8, 15, 16, 17, 64])
        func float32WithinTheBound(rows: Int) {
            KernelLock.run {
                let k = 256
                for n in [64, 1003] {
                    for (aT, bT) in [(false, false), (true, false), (false, true), (true, true)] {
                        let a0 = MLXRandom.normal(
                            aT ? [k, rows] : [rows, k], dtype: .float32,
                            key: MLXRandom.key(UInt64(rows)))
                        let b0 = MLXRandom.normal(
                            bT ? [n, k] : [k, n], dtype: .float32, key: MLXRandom.key(UInt64(n)))
                        let a = aT ? a0.transposed() : a0
                        let b = bT ? b0.transposed() : b0
                        vmlx_cpu_reset_counters()
                        let y = matmul(a, b)
                        #expect(
                            withinSumBound(y, a, b, c: Double(k + 2)),
                            "rows \(rows) n \(n) aT \(aT) bT \(bT)")
                        let split =
                            Highway.enabled && vmlx_cpu_thread_count() > 1 && rows < 16
                            && rows * n * k >= 65536 && n / 64 >= 2
                        #expect(
                            (vmlx_cpu_sgemm_column_splits() > 0) == split,
                            "rows \(rows) n \(n): column split expected \(split)")
                    }
                }
            }
        }

        /// Batched: at least eight batches split across the pool.
        @Test func batchedFloat32() {
            KernelLock.run {
                let a = MLXRandom.normal([16, 8, 64], dtype: .float32, key: MLXRandom.key(1))
                let b = MLXRandom.normal([16, 64, 32], dtype: .float32, key: MLXRandom.key(2))
                #expect(withinSumBound(matmul(a, b), a, b, c: 66))
            }
        }

        /// Native mode's bf16 and fp16 GEMM (#3019's simd_low_precision_gemm.h): the fused GEMV at M = 1
        /// with N*K >= 65536 (B plain and transposed), the converting SGEMM at M*N*K >= 65536, and
        /// simd_gemm below it (its partial blocks and K tail at 3 x 50 x 70); K = 250 and N = 502 leave
        /// tails at 4, 8 and 16 lanes. All accumulate in float32 and round the output once.
        @Test(arguments: [DType.bfloat16, .float16])
        func halfPrecisionWithinTheBound(dtype: DType) {
            KernelLock.run {
                let shapes = [
                    (1, 512, 256), (1, 64, 64), (16, 64, 64), (17, 512, 256), (64, 256, 512),
                    (1, 502, 250),
                    (17, 502, 250), (3, 50, 70),
                ]
                for (rows, n, k) in shapes {
                    for bT in [false, true] {
                        let a = MLXRandom.normal(
                            [rows, k], dtype: .float32, key: MLXRandom.key(UInt64(rows))
                        ).asType(dtype)
                        let b0 = MLXRandom.normal(
                            bT ? [n, k] : [k, n], dtype: .float32, key: MLXRandom.key(UInt64(n))
                        ).asType(dtype)
                        let b = bT ? b0.transposed() : b0
                        let y = matmul(a, b)
                        #expect(y.dtype == dtype)
                        #expect(
                            withinSumBound(y, a, b, c: Double(k + 2)),
                            "\(dtype) \(rows)x\(n)x\(k) bT \(bT)")
                    }
                }
            }
        }
    }
#endif
