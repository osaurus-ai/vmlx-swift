// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX

/// Decides whether an embedder's weights are widened to float32 at load.
///
/// MLX's CPU backend sends float32 matmul to BLAS, but on Linux it runs bfloat16 and float16 matmul
/// through its own scalar code, several times slower (a Granite-shaped body: 0.38 s per 128-token
/// input in float32, 2.28 s in bf16). Until the CPU backend gains a fast dense low-precision path,
/// an embedder that runs on a Linux CPU is therefore loaded in float32. Apple platforms, and builds
/// whose default device is a GPU, are unaffected.
package enum CPUPrecisionPolicy {
    /// Keeps the checkpoint's dtype when set to 1, true, yes or on.
    package static let nativeDTypeVariable = "VMLX_CPU_NATIVE_DTYPE"

    /// Whether this build's CPU backend has a dense bf16/fp16 matmul at least as fast as float32
    /// BLAS. Define `VMLX_CPU_FAST_DENSE_LOW_PRECISION` only once that has been measured on arm64
    /// and x86_64. Apple platforms answer true by choice, which keeps their behaviour unchanged.
    package static var buildHasFastDenseLowPrecision: Bool {
        #if os(Linux) && !VMLX_CPU_FAST_DENSE_LOW_PRECISION
            false
        #else
            true
        #endif
    }

    package static func nativeDTypeRequested(environment: [String: String]) -> Bool {
        RuntimeEnvironment.flag(nativeDTypeVariable, in: environment)
    }

    package static func shouldWiden(
        buildHasFastDenseLowPrecision: Bool, defaultDevice: DeviceType?, nativeDTypeRequested: Bool
    ) -> Bool {
        !buildHasFastDenseLowPrecision && defaultDevice == .cpu && !nativeDTypeRequested
    }

    /// Converts every bfloat16 and float16 tensor to float32, lazily, the scales and biases of
    /// affine-quantized layers included (MLX's quantized matmul accepts float32 ones). The rest
    /// stays as it is: packed quantized weights, integer tensors and uint8 scales.
    package static func widened(_ weights: [String: MLXArray]) -> (
        weights: [String: MLXArray], converted: Int, footprintBytes: Int, addedBytes: Int
    ) {
        var result = weights
        var converted = 0
        var footprint = 0
        var added = 0
        for (key, value) in weights {
            switch value.dtype {
            case .bfloat16, .float16:
                result[key] = value.asType(.float32)
                converted += 1
                footprint += value.size * 4
                added += value.size * 2
            default:
                footprint += value.nbytes
            }
        }
        return (result, converted, footprint, added)
    }

    /// Widening is allowed while the weights would take at most half the memory limit.
    package static func fitsMemoryBudget(footprintBytes: Int, limitBytes: Int) -> Bool {
        footprintBytes <= limitBytes / 2
    }

    /// The smallest of physical memory and every cgroup v2 `memory.max` from this process's cgroup
    /// up to the root, so that containers, systemd services and limits on a parent all count.
    /// cgroup v1 is not read.
    package static func memoryLimitBytes(
        physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory,
        readFile: (String) -> String? = { try? String(contentsOfFile: $0, encoding: .utf8) }
    ) -> Int {
        var limit = physicalMemory
        if let membership = readFile("/proc/self/cgroup")?
            .split(separator: "\n").first(where: { $0.hasPrefix("0::") })
        {
            var components = membership.dropFirst(3).split(separator: "/").map(String.init)
            while true {
                let directory = (["/sys/fs/cgroup"] + components).joined(separator: "/")
                if let raw = readFile(directory + "/memory.max")?
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                    let value = UInt64(raw)
                {
                    limit = min(limit, value)
                }
                if components.isEmpty { break }
                components.removeLast()
            }
        }
        return Int(clamping: limit)
    }

    /// The loader's hook. Returns its input at once where the build has a fast path (all Apple
    /// platforms); otherwise reads the process state and lets `decide` apply the rule.
    package static func applyOnLoad(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        if buildHasFastDenseLowPrecision { return weights }
        return decide(
            weights, buildHasFastDenseLowPrecision: false,
            defaultDevice: Device.defaultDevice().deviceType,
            nativeDTypeRequested: RuntimeEnvironment.flag(nativeDTypeVariable),
            limitBytes: { memoryLimitBytes() },
            report: { FileHandle.standardError.write(Data(($0 + "\n").utf8)) })
    }

    /// The rule with every input injected. `report` receives one line whenever the rule acts: when
    /// it widens, and when the memory guard keeps the checkpoint dtype.
    package static func decide(
        _ weights: [String: MLXArray], buildHasFastDenseLowPrecision: Bool,
        defaultDevice: DeviceType?, nativeDTypeRequested: Bool, limitBytes: () -> Int,
        report: (String) -> Void
    ) -> [String: MLXArray] {
        guard
            shouldWiden(
                buildHasFastDenseLowPrecision: buildHasFastDenseLowPrecision,
                defaultDevice: defaultDevice, nativeDTypeRequested: nativeDTypeRequested)
        else { return weights }

        let widening = widened(weights)
        guard widening.converted > 0 else { return weights }
        let kinds = Set(weights.values.map(\.dtype)).intersection([.bfloat16, .float16])
        let label = kinds == [.float16] ? "fp16" : kinds == [.bfloat16] ? "bf16" : "bf16/fp16"
        let tensors = widening.converted == 1 ? "tensor" : "tensors"

        let limit = limitBytes()
        guard fitsMemoryBudget(footprintBytes: widening.footprintBytes, limitBytes: limit) else {
            report(
                "[Load] cpu-float32: kept \(widening.converted) \(label) \(tensors) as is: "
                    + "\(bytes(widening.footprintBytes)) in float32 would exceed half the memory "
                    + "limit (\(bytes(limit)))")
            return weights
        }
        report(
            "[Load] cpu-float32: \(widening.converted) \(label) \(tensors) -> float32 "
                + "(+\(bytes(widening.addedBytes)), \(bytes(widening.footprintBytes)) total): "
                + "this CPU backend has no fast dense low-precision matmul; "
                + "set \(nativeDTypeVariable)=1 to keep the checkpoint dtype")
        return widening.weights
    }

    /// Binary units, as vmlx's other memory messages use.
    private static func bytes(_ count: Int) -> String {
        count >= 1_073_741_824
            ? String(format: "%.2f GiB", Double(count) / 1_073_741_824)
            : String(format: "%.0f MiB", Double(count) / 1_048_576)
    }
}
