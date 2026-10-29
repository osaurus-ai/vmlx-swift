// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX

/// Keeps MLX's scheduling memory limit at or above what the loaded model and its KV cache hold right now.
///
/// `MLX.Memory.memoryLimit` is a scheduling guideline: while the allocator's active memory is above it, every eval
/// waits for an outstanding task before it schedules more (`eval_impl` in mlx/transforms.cpp: `get_active_memory() >
/// get_memory_limit()` -> `scheduler::wait_for_one()`). That bounds transient temporaries. It cannot free live state:
/// the weights and the KV cache stay allocated however long the queue drains. A limit below the live working set
/// therefore makes every decode step wait without freeing anything. Measured on MiniMax-M3.1 JANGHT1.9 (M5 Ultra,
/// BatchEngine, 35.7k-token prompt): 9.7 tok/s with the 97.3 GiB limit Osaurus' Safe Auto plan applied on a 128 GB Mac
/// vs 36.9 tok/s with MLX's native ceiling, same process, interleaved probes; at 4.5k tokens the two were equal.
///
/// A limit set at load from an estimate cannot know how far the KV will grow, so the generation loops call
/// ``observe()`` at every step boundary: when the active memory plus ``headroomBytes`` exceeds the limit, the limit is
/// raised to that sum (never above MLX's own default ceiling, never lowered here). Transient temporaries above the
/// live set plus the headroom still wait as before. `VMLX_SCHEDULING_FLOOR=0` disables it (A/B only).
public enum SchedulingMemoryFloor {
    /// Scratch kept above the live working set (one long-prefill chunk's temporaries fit with room to spare).
    public static let headroomBytes = 4 << 30

    static let environmentEnabled: Bool =
        ProcessInfo.processInfo.environment["VMLX_SCHEDULING_FLOOR"] != "0"
    /// Tests: on / off for the current task (nil = the environment's).
    @TaskLocal public static var enabledOverride: Bool?
    public static var isEnabled: Bool { enabledOverride ?? environmentEnabled }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var _raises = 0
    nonisolated(unsafe) private static var _lastRaisedTo = 0

    /// How often ``observe()`` raised the limit in this process, and the last value it set (diagnostics, tests).
    public static var raises: Int { lock.withLock { _raises } }
    public static var lastRaisedTo: Int { lock.withLock { _lastRaisedTo } }

    /// MLX's own startup ceiling (allocator.cpp: min(1.5 x the recommended working set, 0.95 x RAM)), the most the
    /// floor ever sets.
    public static func nativeCeiling(
        physicalMemory: Int = Int(clamping: ProcessInfo.processInfo.physicalMemory),
        recommendedWorkingSetBytes: Int? = MLX.GPU.maxRecommendedWorkingSetBytes()
    ) -> Int {
        let physical95 = (physicalMemory / 20) * 19 + ((physicalMemory % 20) * 19) / 20
        let recommended150: Int = {
            guard let recommendedWorkingSetBytes else { return Int.max }
            let half = recommendedWorkingSetBytes / 2
            guard recommendedWorkingSetBytes <= Int.max - half else { return Int.max }
            return recommendedWorkingSetBytes + half
        }()
        return max(1 << 30, min(physical95, recommended150))
    }

    /// The limit the floor wants for an active-memory reading, or nil when the current limit already covers it.
    /// Pure arithmetic (tests).
    static func raisedLimit(active: Int, limit: Int, ceiling: Int, headroom: Int = headroomBytes) -> Int? {
        guard active > 0, limit > 0 else { return nil }
        let (want, overflow) = active.addingReportingOverflow(headroom)
        let target = overflow ? ceiling : min(want, ceiling)
        return target > limit ? target : nil
    }

    /// Raise MLX's scheduling limit to the live working set + headroom when it sits below. Cheap (two allocator
    /// reads); called once per decode step / scheduler step.
    @discardableResult
    public static func observe() -> Bool {
        guard isEnabled else { return false }
        let limit = MLX.Memory.memoryLimit
        guard let target = raisedLimit(
            active: MLX.Memory.activeMemory, limit: limit, ceiling: nativeCeiling())
        else { return false }
        MLX.Memory.memoryLimit = target
        let first: Bool = lock.withLock {
            _raises += 1
            _lastRaisedTo = target
            return _raises == 1
        }
        if first || RuntimeReceipts.isEnabled {
            RuntimeReceipts.emit(
                "scheduling-floor limit_gib=\(RuntimeReceipts.gib(limit)) -> \(RuntimeReceipts.gib(target)) "
                    + "active_gib=\(RuntimeReceipts.gib(MLX.Memory.activeMemory))")
        }
        return true
    }
}

/// One-line runtime receipts for live proofs (`VMLX_RUNTIME_RECEIPTS=1`): `[vmlx-receipt] <kind> key=value ...` through
/// NSLog (stderr + the unified log), the same channel as the `[dSpark]` stats lines. Off by default; nothing on the
/// hot path when off.
public enum RuntimeReceipts {
    public static let isEnabled: Bool =
        ProcessInfo.processInfo.environment["VMLX_RUNTIME_RECEIPTS"] == "1"

    private static let lock = NSLock()
    nonisolated(unsafe) private static var counter = 0

    /// A process-wide request number for correlating one request's receipts.
    public static func nextRequestID() -> Int {
        lock.withLock {
            counter += 1
            return counter
        }
    }

    /// Set by the generation loop around iterator construction so the iterator's receipts carry the loop's number.
    @TaskLocal public static var currentRequestID: Int?

    public static func emit(_ line: String) {
        guard isEnabled else { return }
        NSLog("%@", "[vmlx-receipt] " + line)
    }

    /// The thinking mode a request rendered with, read from its cache scope salt (`cacheScopeSalt(from:)` writes
    /// `thinking_mode=<mode>` for MiniMax-M3 after `MiniMaxM3ThinkingTemplateContext` mapped the toggle, and
    /// `reasoning=on|off` for families that keep `enable_thinking`). No mode = the template's default (adaptive on M3).
    public static func thinkingMode(fromScopeSalt salt: String?) -> String {
        let parts = (salt ?? "").split(whereSeparator: { $0 == "|" || $0 == ";" || $0 == "," || $0 == " " })
        for part in parts where part.hasPrefix("thinking_mode=") {
            return String(part.dropFirst("thinking_mode=".count))
        }
        for part in parts where part.hasPrefix("reasoning=") {
            return "toggle_" + String(part.dropFirst("reasoning=".count))
        }
        return "adaptive"
    }

    public static func draftName(_ strategy: DraftStrategy?) -> String {
        guard let strategy else { return "none" }
        switch strategy {
        case .none: return "none"
        default:
            let text = String(describing: strategy)
            return String(text.prefix(while: { $0 != "(" }))
        }
    }

    public static func gib(_ bytes: Int) -> String {
        String(format: "%.2f", Double(bytes) / 1_073_741_824)
    }
}
