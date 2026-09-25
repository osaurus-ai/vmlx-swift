// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
import Foundation

/// Opt-in wall-clock attribution; never evaluates or synchronizes MLX arrays.
struct CacheFinalizationTrace {
    private static let enabled = ProcessInfo.processInfo.environment["VMLX_CACHE_FETCH_TRACE"] == "1"
    private let operation: String
    private let tokens: Int
    private let start: UInt64
    private var previous: UInt64
    init(_ operation: String, tokens: Int) {
        self.operation = operation
        self.tokens = tokens
        self.start = Self.enabled ? DispatchTime.now().uptimeNanoseconds : 0
        self.previous = self.start
    }
    mutating func mark(_ phase: String) {
        guard Self.enabled else { return }
        let now = DispatchTime.now().uptimeNanoseconds
        let delta = Double(now - previous) / 1e6
        let total = Double(now - start) / 1e6
        previous = now
        FileHandle.standardError.write(Data(
            "[vmlx][cache/finalize-phase] operation=\(operation) tokens=\(tokens) phase=\(phase) deltaMs=\(delta) totalMs=\(total)\n".utf8))
    }
}
