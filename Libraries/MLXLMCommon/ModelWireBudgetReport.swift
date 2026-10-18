// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX

/// What the loader learned about the system wire budget for the model it just loaded.
///
/// Root cause this exists for (2026-10-15): Apple GPUs need every buffer a command buffer touches to be
/// WIRED, and macOS caps total wired memory (`vm.global_user_wire_limit`). Allosaurus with adaptive MTP
/// keeps ~37.6 GiB wired during decode; a 48 GB Mac (wire cap 37.92 GiB = 79%) leaves ~30.9 GiB of wire headroom after macOS.s own
/// wiring. Every command buffer then fights the kernel for wiring and decode fell to 0.2 tok/s on a user's
/// M4 Pro — with plenty of free RAM and a green memory-pressure graph, so nothing in the UI explained it.
///
/// This report is ADVISORY ONLY. It never blocks or changes a load (Eric's rule: estimates advise, never
/// refuse). Hosts surface ``advisory`` so the user sees the real constraint and their options, and sample
/// ``SystemWireBudget/snapshot()`` before/after a turn to report `wireLimitRejections` for that turn.
public struct ModelWireBudgetReport: Sendable, Equatable {
    /// GPU-resident weight bytes (SSD-only tables excluded).
    public let modelBytes: Int
    /// Estimated wired bytes during generation: weights + measured KV/scratch/reuse overhead.
    public let estimatedGenerationBytes: Int
    /// The budget as observed at load, before this model's residency was applied.
    public let budget: SystemWireBudget
    /// The residency limit applied, or `nil` when the model could not be pinned.
    public let residencyTargetBytes: Int?
    /// Installed RAM the load's memory policy used (``HostPhysicalMemory``).
    public let physicalMemoryBytes: Int

    public init(
        modelBytes: Int, budget: SystemWireBudget, residencyTargetBytes: Int?,
        physicalMemoryBytes: Int = Int(clamping: HostPhysicalMemory.bytes)
    ) {
        self.modelBytes = modelBytes
        self.estimatedGenerationBytes = modelBytes + SystemWireBudget.estimatedGenerationOverheadBytes
        self.budget = budget
        self.residencyTargetBytes = residencyTargetBytes
        self.physicalMemoryBytes = physicalMemoryBytes
    }

    /// Bytes the generation is expected to need beyond what can still be wired; `nil` when unknown.
    public var shortfallBytes: Int? {
        guard let headroom = budget.headroomBytes else { return nil }
        return max(0, estimatedGenerationBytes - headroom)
    }

    /// `VMLX_WIRE_BUDGET_CACHE_CEILING=0` disables the ceiling (same-binary A/B and diagnosis).
    static let allocatorCacheCeilingEnabled: Bool = {
        let value = RuntimeEnvironment.value("VMLX_WIRE_BUDGET_CACHE_CEILING") ?? "1"
        return value != "0" && value.lowercased() != "false"
    }()

    /// Smallest MLX freed-buffer pool a ceiling may impose: below this, decode re-allocates (and re-wires) its
    /// intermediates every step.
    public static let minimumAllocatorCacheCeilingBytes = 512 * 1024 * 1024

    /// Largest MLX freed-buffer pool (`Memory.cacheLimit`) that keeps a generation inside this Mac's wire budget,
    /// or `nil` when there is no such constraint to apply (budget unknown, or a Mac of 64 GiB or more).
    ///
    /// SPEED-CRITICAL (Allosaurus 0.2 on 48 GB Macs, 2026-10-18). The GPU can only use WIRED memory, and the
    /// allocator's freed-buffer pool is wired just like weights and KV. Hosts sized the pool from RAM and weights
    /// only (Osaurus: 6 GB on a 48 GB Mac while MTP runs), so weights + KV + scratch + pool overran the ~31 GiB a
    /// 48 GB Mac can wire: the kernel refused wiring on every command buffer (`vm.add_wire_count_over_global_limit`)
    /// and decode fell from 46-64 to 23-34 tok/s, MTP on or off. The pool must fit what is left of the wire
    /// budget (and of the residency set, whose overflow is re-wired per command buffer) after the generation's
    /// own working set. Advisory sizing of a cache only: it never blocks or changes a load.
    public var allocatorCacheCeilingBytes: Int? {
        guard Self.allocatorCacheCeilingEnabled, physicalMemoryBytes < 64 * 1_073_741_824,
            let headroom = budget.headroomBytes
        else { return nil }
        let wireable = min(headroom, residencyTargetBytes ?? headroom)
        return max(Self.minimumAllocatorCacheCeilingBytes, wireable - estimatedGenerationBytes)
    }

    /// `true` when the estimated generation working set exceeds the measured wire headroom.
    public var exceedsWireBudget: Bool { (shortfallBytes ?? 0) > 0 }

    /// User-facing explanation when the model cannot stay fully wired. `nil` when it fits or is unknown.
    public var advisory: String? {
        guard exceedsWireBudget, let headroom = budget.headroomBytes, let short = shortfallBytes else {
            return nil
        }
        let gib = { (b: Int) in String(format: "%.1f GB", Double(b) / 1_073_741_824) }
        return """
            This model needs about \(gib(estimatedGenerationBytes)) of GPU-pinned memory while generating, \
            but macOS currently allows about \(gib(headroom)) more to be pinned (\(gib(short)) short). \
            Replies may be slower than normal. Options: use a smaller model, turn off speculative \
            decoding (MTP) and reload the model (its draft head stays pinned until a reload), close \
            GPU-heavy apps, or raise the macOS wired limits as an administrator (resets at reboot). \
            The model still loads.
            """
    }

    /// One stable log line for support bundles and RunBench receipts.
    public var logLine: String {
        func v(_ x: Int?) -> String { x.map(String.init) ?? "unknown" }
        return "[Load] wire_budget model_bytes=\(modelBytes) est_generation_bytes=\(estimatedGenerationBytes) "
            + "user_wire_limit=\(v(budget.userWireLimitBytes)) gpu_wired_limit=\(v(budget.gpuWiredLimitBytes)) "
            + "wired_at_load=\(v(budget.wiredBytes)) headroom=\(v(budget.headroomBytes)) "
            + "shortfall=\(v(shortfallBytes)) residency_target=\(v(residencyTargetBytes)) "
            + "allocator_cache_ceiling=\(v(allocatorCacheCeilingBytes)) "
            + "wire_rejections_total=\(v(budget.wireLimitRejections))"
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var _lastLoad: ModelWireBudgetReport?

    /// The report from the most recent model load in this process.
    public static var lastLoad: ModelWireBudgetReport? {
        lock.lock(); defer { lock.unlock() }
        return _lastLoad
    }

    static func recordLastLoad(_ report: ModelWireBudgetReport) {
        lock.lock(); defer { lock.unlock() }
        _lastLoad = report
    }
}
