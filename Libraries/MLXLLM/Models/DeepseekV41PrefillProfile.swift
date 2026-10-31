// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 prefill component profile (diagnostic only, off by default).
//
// While `active`, every `mark` evaluates the arrays it is given, waits for the GPU and charges the time since the
// previous mark to `<group>/<name>`. The arithmetic is unchanged; the forced evaluations change scheduling only, so
// an instrumented prefill is slower than an uninstrumented one and the sum is checked against the uninstrumented
// wall by the caller (RunBench `BENCH_DSV41_PREFILL`, phase `profile`). `VMLX_DSV41_PREFILL_PROFILE=1` arms it for
// the whole process. Inactive, a mark is one flag test.

import Foundation
import MLX
import MLXLMCommon

public enum DeepseekV41PrefillProfile {
    nonisolated(unsafe) public static var active: Bool =
        ProcessInfo.processInfo.environment["VMLX_DSV41_PREFILL_PROFILE"] == "1"
    /// Prefix of the next marks (the model sets the layer group).
    nonisolated(unsafe) public static var group = "model"
    nonisolated(unsafe) public private(set) static var seconds: [String: Double] = [:]
    nonisolated(unsafe) public private(set) static var counts: [String: Int] = [:]
    nonisolated(unsafe) private static var last = CFAbsoluteTimeGetCurrent()

    public static func reset() {
        seconds = [:]
        counts = [:]
        group = "model"
        last = CFAbsoluteTimeGetCurrent()
    }

    /// Restart the clock without charging anyone (after evaluating `arrays`), e.g. at a forward's start.
    static func begin(_ arrays: @autoclosure () -> [MLXArray]) {
        guard active, !CompiledDecodeTrace.isActive else { return }
        MLX.eval(arrays())
        last = CFAbsoluteTimeGetCurrent()
    }

    static func mark(_ name: String, _ arrays: @autoclosure () -> [MLXArray]) {
        guard active, !CompiledDecodeTrace.isActive else { return }
        MLX.eval(arrays())
        let now = CFAbsoluteTimeGetCurrent()
        let key = group + "/" + name
        seconds[key, default: 0] += now - last
        counts[key, default: 0] += 1
        last = now
    }
}
