// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import Testing

/// The float32 rule's decisions, on every platform. Some tests build arrays, so on macOS this suite
/// needs the metallib staged next to the test binary.
@Suite(.serialized) struct CPUPrecisionPolicyTests {

    @Test(arguments: ["1", "true", "yes", "on", "TRUE", " on "])
    func overrideAcceptsTheFlagSpellings(_ value: String) {
        #expect(
            CPUPrecisionPolicy.nativeDTypeRequested(environment: ["VMLX_CPU_NATIVE_DTYPE": value]))
    }

    /// Catches a misspelled variable name: only the exact name counts.
    @Test func overrideIgnoresOtherValuesAndNames() {
        #expect(!CPUPrecisionPolicy.nativeDTypeRequested(environment: [:]))
        #expect(
            !CPUPrecisionPolicy.nativeDTypeRequested(environment: ["VMLX_CPU_NATIVE_DTYPE": "0"]))
        #expect(!CPUPrecisionPolicy.nativeDTypeRequested(environment: ["VMLX_CPU_NATIVE": "1"]))
        #expect(!CPUPrecisionPolicy.nativeDTypeRequested(environment: ["VMLX_NATIVE_DTYPE": "1"]))
    }

    /// Of the eight combinations, exactly one widens: no fast path, CPU device, no override.
    @Test func widensForExactlyOneCombination() {
        var widening: [(Bool, DeviceType, Bool)] = []
        for fast in [false, true] {
            for device in [DeviceType.cpu, .gpu] {
                for requested in [false, true]
                where CPUPrecisionPolicy.shouldWiden(
                    buildHasFastDenseLowPrecision: fast, defaultDevice: device,
                    nativeDTypeRequested: requested)
                {
                    widening.append((fast, device, requested))
                }
            }
        }
        #expect(widening.count == 1)
        #expect(widening.first?.0 == false)
        #expect(widening.first?.1 == .cpu)
        #expect(widening.first?.2 == false)
        #expect(
            !CPUPrecisionPolicy.shouldWiden(
                buildHasFastDenseLowPrecision: false, defaultDevice: nil,
                nativeDTypeRequested: false))
    }

    @Test func widensOnlyFloatingTensorsAndCountsBytes() {
        let weights: [String: MLXArray] = [
            "bf16": MLXArray([1.5, -2.25, 3, 4, 5, 6] as [Float]).reshaped(2, 3).asType(.bfloat16),
            "f16": MLXArray([0.5, 1, 2, 4] as [Float]).asType(.float16),
            "f32": MLXArray([1, 2, 3, 4, 5] as [Float]),
            "packed": MLXArray([1, 2] as [UInt32]),
            "scales8": MLXArray([1, 2, 3] as [UInt8]),
            "ids": MLXArray([7] as [Int32]),
        ]
        let result = CPUPrecisionPolicy.widened(weights)
        // The dtype checks catch a missing conversion; asArray alone would convert bf16 by itself.
        #expect(result.weights["bf16"]?.dtype == .float32)
        #expect(result.weights["bf16"]?.shape == [2, 3])
        #expect(result.weights["bf16"]?.asArray(Float.self) == [1.5, -2.25, 3, 4, 5, 6])
        #expect(result.weights["f16"]?.dtype == .float32)
        #expect(result.weights["f32"]?.dtype == .float32)
        #expect(result.weights["packed"]?.dtype == .uint32)
        #expect(result.weights["scales8"]?.dtype == .uint8)
        #expect(result.weights["ids"]?.dtype == .int32)
        #expect(result.converted == 2)
        // after widening: bf16 6×4 + f16 4×4 + f32 5×4 + uint32 2×4 + uint8 3×1 + int32 1×4 = 75
        #expect(result.footprintBytes == 75)
        // added: (6 + 4) elements × 2 bytes
        #expect(result.addedBytes == 20)
    }

    @Test func memoryBudgetIsHalfTheLimit() {
        #expect(CPUPrecisionPolicy.fitsMemoryBudget(footprintBytes: 50, limitBytes: 100))
        #expect(!CPUPrecisionPolicy.fitsMemoryBudget(footprintBytes: 51, limitBytes: 100))
        #expect(CPUPrecisionPolicy.fitsMemoryBudget(footprintBytes: 10, limitBytes: 100))
    }

    @Test func memoryLimitIsTheSmallestOnTheCgroupPath() {
        let files = [
            "/proc/self/cgroup": "0::/system.slice/app.service\n",
            "/sys/fs/cgroup/system.slice/app.service/memory.max": "max\n",
            "/sys/fs/cgroup/system.slice/memory.max": "4000\n",
        ]
        #expect(
            CPUPrecisionPolicy.memoryLimitBytes(physicalMemory: 10_000, readFile: { files[$0] })
                == 4000)
        #expect(
            CPUPrecisionPolicy.memoryLimitBytes(physicalMemory: 3000, readFile: { files[$0] })
                == 3000)
        // The process's own cgroup has a numeric limit, below its parent's.
        let ownLimit = files.merging(
            ["/sys/fs/cgroup/system.slice/app.service/memory.max": "1500\n"]) { $1 }
        #expect(
            CPUPrecisionPolicy.memoryLimitBytes(physicalMemory: 10_000, readFile: { ownLimit[$0] })
                == 1500)
        // A hybrid v1/v2 host: the v1 lines are skipped, and the "0::" line is read as above.
        let hybrid = files.merging(
            ["/proc/self/cgroup": "12:memory:/x\n0::/system.slice/app.service\n"]) { $1 }
        #expect(
            CPUPrecisionPolicy.memoryLimitBytes(physicalMemory: 10_000, readFile: { hybrid[$0] })
                == 4000)
        // A container's private cgroup namespace: the process sits at the root, whose memory.max
        // is the container's limit.
        let container = ["/proc/self/cgroup": "0::/\n", "/sys/fs/cgroup/memory.max": "2000\n"]
        #expect(
            CPUPrecisionPolicy.memoryLimitBytes(physicalMemory: 10_000, readFile: { container[$0] })
                == 2000)
        // cgroup v1 only: no "0::" line, so only physical memory counts.
        #expect(
            CPUPrecisionPolicy.memoryLimitBytes(
                physicalMemory: 5000,
                readFile: { $0 == "/proc/self/cgroup" ? "4:memory:/x\n" : nil })
                == 5000)
    }

    static func bf16Weights() -> [String: MLXArray] {
        [
            "a": MLXArray([1, 2, 3, 4] as [Float]).asType(.bfloat16),
            "b": MLXArray([1, 2] as [UInt32]),
        ]
    }

    @Test func decideWidensAndReportsWhenTheBudgetAllows() {
        var lines: [String] = []
        let out = CPUPrecisionPolicy.decide(
            Self.bf16Weights(), buildHasFastDenseLowPrecision: false, defaultDevice: .cpu,
            nativeDTypeRequested: false, limitBytes: { 1_000_000 }, report: { lines.append($0) })
        #expect(out["a"]?.dtype == .float32)
        #expect(out["b"]?.dtype == .uint32)
        #expect(lines.count == 1)
        #expect(lines.first?.hasPrefix("[Load] cpu-float32: 1 bf16 tensor -> float32 (+") == true)
        #expect(lines.first?.contains("set VMLX_CPU_NATIVE_DTYPE=1") == true)
    }

    /// Nothing to widen: the rule is on, but the weights come back as they were, unreported.
    @Test func decideLeavesAFloat32CheckpointAlone() {
        var lines: [String] = []
        let weights: [String: MLXArray] = [
            "a": MLXArray([1, 2, 3, 4] as [Float]),
            "b": MLXArray([1, 2] as [UInt32]),
        ]
        let out = CPUPrecisionPolicy.decide(
            weights, buildHasFastDenseLowPrecision: false, defaultDevice: .cpu,
            nativeDTypeRequested: false, limitBytes: { 1_000_000 }, report: { lines.append($0) })
        #expect(out.count == 2)
        #expect(out["a"] === weights["a"])
        #expect(out["b"] === weights["b"])
        #expect(lines.isEmpty)
    }

    /// 512 × 1024 fp16 values: 1 MiB, and 2 MiB once widened. `decide` evaluates nothing, so the
    /// arrays in this test and the next never allocate their data.
    @Test func decideReportsAnFP16Checkpoint() {
        var lines: [String] = []
        _ = CPUPrecisionPolicy.decide(
            ["w": MLXArray.zeros([512, 1024], dtype: .float16)],
            buildHasFastDenseLowPrecision: false, defaultDevice: .cpu, nativeDTypeRequested: false,
            limitBytes: { 1 << 40 }, report: { lines.append($0) })
        #expect(lines.count == 1)
        #expect(
            lines.first?.hasPrefix(
                "[Load] cpu-float32: 1 fp16 tensor -> float32 (+1 MiB, 2 MiB total): ") == true)
    }

    /// 1 GiB of fp16 and 1 GiB of bf16: 4 GiB once widened.
    @Test func decideReportsAMixedCheckpoint() {
        var lines: [String] = []
        _ = CPUPrecisionPolicy.decide(
            [
                "h": MLXArray.zeros([1 << 29], dtype: .float16),
                "b": MLXArray.zeros([1 << 29], dtype: .bfloat16),
            ],
            buildHasFastDenseLowPrecision: false, defaultDevice: .cpu, nativeDTypeRequested: false,
            limitBytes: { 1 << 40 }, report: { lines.append($0) })
        #expect(lines.count == 1)
        #expect(
            lines.first?.hasPrefix(
                "[Load] cpu-float32: 2 bf16/fp16 tensors -> float32 (+2.00 GiB, 4.00 GiB total): ")
                == true)
    }

    /// The memory guard: 4×4 + 2×4 = 24 bytes after widening exceeds half of a 40-byte limit.
    @Test func decideKeepsTheCheckpointDTypeOverTheBudget() {
        var lines: [String] = []
        let out = CPUPrecisionPolicy.decide(
            Self.bf16Weights(), buildHasFastDenseLowPrecision: false, defaultDevice: .cpu,
            nativeDTypeRequested: false, limitBytes: { 40 }, report: { lines.append($0) })
        #expect(out["a"]?.dtype == .bfloat16)
        #expect(lines.count == 1)
        #expect(lines.first?.hasPrefix("[Load] cpu-float32: kept 1 bf16 tensor as is: ") == true)
    }

    @Test func decideDoesNothingWhenTheRuleIsOff() {
        var lines: [String] = []
        for (fast, device, requested) in [
            (true, DeviceType.cpu, false), (false, DeviceType.gpu, false),
            (false, DeviceType.cpu, true),
        ] {
            let out = CPUPrecisionPolicy.decide(
                Self.bf16Weights(), buildHasFastDenseLowPrecision: fast, defaultDevice: device,
                nativeDTypeRequested: requested, limitBytes: { 1_000_000 },
                report: { lines.append($0) })
            #expect(out["a"]?.dtype == .bfloat16)
        }
        #expect(lines.isEmpty)
    }

    #if !os(Linux)
        /// Apple platforms never widen, whatever the environment says.
        @Test func applePlatformsAreUntouched() {
            #expect(CPUPrecisionPolicy.buildHasFastDenseLowPrecision)
            #expect(CPUPrecisionPolicy.applyOnLoad(Self.bf16Weights())["a"]?.dtype == .bfloat16)
        }
    #endif
}
