// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#if os(Linux)
    import CmlxCPUShim
    import Foundation
    import MLX
    import Testing

    @Suite(.serialized) struct TargetsTests {
        /// The dispatched kernels execute exactly the targets the runner expects for this host, so a
        /// target dropped from dispatch cannot pass by shrinking the expectation.
        @Test func executedTargetsAreThisHostsExpectation() throws {
            try KernelLock.run {
                let expectation = ProcessInfo.processInfo.environment["VMLX_EXPECT_HWY_TARGETS"]
                guard Highway.enabled else {
                    #expect(expectation == nil || expectation == "none")
                    return
                }
                let expected = try #require(
                    expectation, "run through scripts/run-cpu-kernel-tests.sh, which sets it")
                // RMSNorm dispatches a float32 kernel at every target.
                let x = MLXRandom.normal([4, 512], dtype: .float32, key: MLXRandom.key(7))
                let weight = MLXArray.ones([512], dtype: .float32)
                let ran = forEachTarget(VMLX_CPU_FAMILY_RMS_NORM) { _ in
                    _ = MLXFast.rmsNorm(x, weight: weight, eps: 1e-5).sum().item(Float.self)
                }
                #expect(Set(ran) == Set(expected.split(separator: " ").map(String.init)))
            }
        }

        /// What the shim reports agrees with itself, and the int8 switch starts off: its default,
        /// since the runner unsets MLX_CPU_QUANTIZED_INT8.
        @Test func shimAnswersAgree() {
            KernelLock.run {
                #expect(vmlx_cpu_thread_count() >= 1)
                #expect(String(cString: vmlx_cpu_thread_setting_error()).isEmpty)
                #expect(!KernelLock.initialQuantizedInt8, "the int8 switch started on")
                if Highway.enabled {
                    #expect(Highway.compiled != 0)
                    #expect(Highway.compiled & Highway.supported != 0)
                    // vmlx links OpenBLAS on Linux, and the pool pins it at every size, 1 included.
                    #expect(vmlx_cpu_openblas_pinned())
                } else {
                    #expect(Highway.compiled == 0)
                    #expect(String(cString: vmlx_cpu_thread_count_reason()) == "no pool")
                }
            }
        }
    }
#endif
