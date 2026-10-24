// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX

/// Trace-only (`VMLX_CACHE_FETCH_TRACE=1`): MLX active / allocator-cache / peak memory at named
/// generation stages, so a footprint peak can be attributed to the stage that created it.
/// The peak is reset after each line, so each line's peak covers the work since the previous one.
enum MemoryStageTrace {
    static let enabled = ProcessInfo.processInfo.environment["VMLX_CACHE_FETCH_TRACE"] == "1"

    static func log(_ stage: String, tokens: Int, detail: String = "") {
        guard enabled else { return }
        let gb = { (bytes: Int) in String(format: "%.2f", Double(bytes) / 1e9) }
        RuntimeDiagnostics.write(Data(
            ("[vmlx][memory/stage] stage=\(stage) tokens=\(tokens) active=\(gb(MLX.Memory.activeMemory))GB "
                + "cache=\(gb(MLX.Memory.cacheMemory))GB peakSincePrevious=\(gb(MLX.Memory.peakMemory))GB"
                + (detail.isEmpty ? "" : " detail=\(detail)") + "\n").utf8))
        MLX.Memory.peakMemory = 0
    }
}
