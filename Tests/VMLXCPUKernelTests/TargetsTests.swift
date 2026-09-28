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
        /// since the runner unsets MLX_CPU_QUANTIZED_INT8. On Linux the shim reads back the core's
        /// own switch, with Highway kernels or without.
        @Test func shimAnswersAgree() {
            KernelLock.run {
                #expect(vmlx_cpu_thread_count() >= 1)
                #expect(String(cString: vmlx_cpu_thread_setting_error()).isEmpty)
                #expect(!KernelLock.initialQuantizedInt8, "the int8 switch started on")
                vmlx_cpu_set_quantized_int8(true)
                #expect(vmlx_cpu_quantized_int8(), "the int8 switch did not read back on")
                vmlx_cpu_set_quantized_int8(false)
                #expect(!vmlx_cpu_quantized_int8(), "the int8 switch did not read back off")
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

        /// The shim refuses what it cannot serve: a restriction to a target this CPU lacks, whose
        /// dispatch would fault, and a family outside the enum, which has no counters.
        @Test func shimRefusesWhatItCannotServe() {
            KernelLock.run {
                let own = Highway.supported
                for bit in Highway.bits(~own) {
                    #expect(
                        !vmlx_cpu_highway_set_targets_for_test(bit),
                        "accepted \(Highway.name(bit)) (\(bit)), which this CPU lacks")
                }
                #expect(vmlx_cpu_highway_supported_targets() == own, "a refusal left a restriction")
                for bit in Highway.bits(own & Highway.compiled) {
                    #expect(vmlx_cpu_highway_set_targets_for_test(bit), "\(Highway.name(bit))")
                }
                #expect(vmlx_cpu_highway_set_targets_for_test(0))
                // After kernels ran: an RMSNorm, and an fp32 GEMM short enough to split by columns.
                vmlx_cpu_reset_counters()
                let x = MLXRandom.normal([4, 512], dtype: .float32, key: MLXRandom.key(7))
                _ = MLXFast.rmsNorm(x, weight: MLXArray.ones([512]), eps: 1e-5).sum().item(
                    Float.self)
                let a = MLXRandom.normal([8, 256], dtype: .float32, key: MLXRandom.key(8))
                let b = MLXRandom.normal([256, 1003], dtype: .float32, key: MLXRandom.key(9))
                _ = matmul(a, b).sum().item(Float.self)
                #expect(vmlx_cpu_executed_targets(VMLX_CPU_FAMILY_COUNT) == 0)
                #expect(vmlx_cpu_highway_calls(VMLX_CPU_FAMILY_COUNT) == 0)
                #expect(vmlx_cpu_fallback_calls(VMLX_CPU_FAMILY_COUNT) == 0)
            }
        }
    }
#endif
