// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// Opt-in monotonic load-stage timings. Stages measure wall time, including
/// awaited work; they do not claim CPU time or physical SSD throughput.
public struct ModelLoadTiming {
    private let enabled: Bool
    private let started: TimeInterval
    private var previous: TimeInterval
    private let scope: String

    public init(scope: String) {
        enabled = RuntimeEnvironment.flag("VMLX_MODEL_LOAD_TRACE")
        started = ProcessInfo.processInfo.systemUptime
        previous = started
        self.scope = scope
    }

    public mutating func mark(_ stage: String) {
        guard enabled else { return }
        let now = ProcessInfo.processInfo.systemUptime
        LoadDiagnostics.write(Data(
            String(format: "[LoadTiming] scope=%@ stage=%@ elapsed_ms=%.3f total_ms=%.3f\n",
                   scope, stage, (now - previous) * 1000, (now - started) * 1000).utf8))
        previous = now
    }
}
