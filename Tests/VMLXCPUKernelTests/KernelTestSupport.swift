// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

#if os(Linux)
    import CmlxCPUShim
    import Foundation
    import MLX
    import Testing

    /// One lock for the whole target. The Highway target mask, the int8 switch and the dispatch
    /// counters are process-wide, so no two tests may use them at once: every test body runs inside
    /// `KernelLock.run`.
    enum KernelLock {
        private static let lock = NSLock()

        /// Also puts the process-wide state back to its defaults on the way in and out, so that a
        /// test that fails midway cannot leave the int8 switch or a target mask to the next.
        static func run<R>(_ body: () throws -> R) rethrows -> R {
            lock.lock()
            reset()
            defer {
                reset()
                lock.unlock()
            }
            return try body()
        }

        private static func reset() {
            vmlx_cpu_highway_set_targets_for_test(0)
            vmlx_cpu_set_quantized_int8(false)
        }
    }

    /// This process's Highway, through the shim.
    enum Highway {
        static var enabled: Bool { vmlx_cpu_highway_enabled() }
        static var compiled: Int64 { vmlx_cpu_highway_compiled_targets() }

        /// The CPU's own supported set: clears any test restriction first.
        static var supported: Int64 {
            vmlx_cpu_highway_set_targets_for_test(0)
            return vmlx_cpu_highway_supported_targets()
        }

        static func name(_ target: Int64) -> String {
            String(cString: vmlx_cpu_highway_target_name(target))
        }

        static func bits(_ set: Int64) -> [Int64] {
            (0 ..< 63).map { Int64(1) << Int64($0) }.filter { set & $0 != 0 }
        }

        /// Every target this build compiled and this CPU runs.
        static var runnable: [Int64] { bits(compiled & supported) }
    }

    /// Runs `body` once per runnable target, with dispatch restricted to that target, and checks
    /// that `family`'s kernels executed exactly that target and never fell back. Without Highway it
    /// runs `body` once, on the scalar code. Returns the names of the targets that ran.
    @discardableResult
    func forEachTarget(
        _ family: vmlx_cpu_family, sourceLocation: SourceLocation = #_sourceLocation,
        _ body: (String) throws -> Void
    ) rethrows -> [String] {
        guard Highway.enabled else {
            try body("scalar")
            return ["scalar"]
        }
        defer { vmlx_cpu_highway_set_targets_for_test(0) }
        var ran: [String] = []
        for target in Highway.runnable {
            vmlx_cpu_highway_set_targets_for_test(target)
            vmlx_cpu_reset_counters()
            try body(Highway.name(target))
            let executed = vmlx_cpu_executed_targets(family)
            #expect(
                executed == target,
                "restricted to \(Highway.name(target)), executed \(Highway.bits(executed).map(Highway.name))",
                sourceLocation: sourceLocation)
            #expect(vmlx_cpu_highway_calls(family) > 0, sourceLocation: sourceLocation)
            #expect(vmlx_cpu_fallback_calls(family) == 0, sourceLocation: sourceLocation)
            ran.append(Highway.name(target))
        }
        #expect(
            !ran.isEmpty, "no target this build compiled is runnable here",
            sourceLocation: sourceLocation)
        return ran
    }

    /// Distance in units in the last place, through the ordered bit patterns. 0 for equal values
    /// (+0 and -0 included) and for two NaNs; Int.max when exactly one is NaN.
    func ulpDistance(_ a: Float, _ b: Float) -> Int {
        if a.isNaN || b.isNaN { return a.isNaN && b.isNaN ? 0 : Int.max }
        func ordered(_ x: Float) -> Int64 {
            let bits = Int64(Int32(bitPattern: x.bitPattern))
            return bits < 0 ? Int64(Int32.min) - bits : bits
        }
        return Int(abs(ordered(a) - ordered(b)))
    }

    func ulpDistance(_ a: Double, _ b: Double) -> Int {
        if a.isNaN || b.isNaN { return a.isNaN && b.isNaN ? 0 : Int.max }
        func ordered(_ x: Double) -> Int64 {
            let bits = Int64(bitPattern: x.bitPattern)
            return bits < 0 ? Int64.min &- bits : bits
        }
        let (difference, overflow) = ordered(a).subtractingReportingOverflow(ordered(b))
        if overflow || difference == Int64.min { return Int.max }
        return Int(abs(difference))
    }

    /// Deterministic test data: splitmix64, uniform in [low, high]. (low + (high - low) * f with f
    /// just below 1 can round to high.)
    struct SeededRandom {
        private var state: UInt64

        init(seed: UInt64) { state = seed }

        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }

        mutating func floats(_ count: Int, _ low: Float, _ high: Float) -> [Float] {
            (0 ..< count).map { _ in
                low + (high - low) * Float(next() >> 40) / Float(1 << 24)
            }
        }
    }

    /// An MLXArray's values as Doubles on the host: exact for every floating-point dtype (integers
    /// above 2^24 in magnitude would round through Float).
    func doubles(_ array: MLXArray) -> [Double] {
        array.dtype == .float64
            ? array.asArray(Double.self)
            : array.asType(.float32).asArray(Float.self).map(Double.init)
    }
#endif
