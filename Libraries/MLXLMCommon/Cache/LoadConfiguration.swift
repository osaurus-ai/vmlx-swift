// Copyright © 2026 Jinho Jang. All rights reserved.
//
// LoadConfiguration — typed, Sendable surface for cold-tier (MLXPress)
// + resident-cap policy. Designed so osaurus / JANG Studio / CLI tools
// can pin behavior from settings without touching env vars.
//
// Env-var precedence is preserved as a fallback so headless CLI
// invocations keep working: explicit value > env > .auto threshold.
//
// See docs/MLXPRESS.md for the current wiring guide. Older JANGPRESS docs
// record historical design evidence and remain useful during migration.

import Foundation
import MLX

/// Cold-weight (axis E, MLXPress) policy. The driver picks a concrete
/// `MLXPressLoadOptions` from this at load-entry time.
///
/// Three states:
///
/// - ``disabled`` — never instantiate the MLXPress tier; the loader
///   takes the legacy full-resident path. Use this when you want
///   strict byte-compat with pre-iter-23 behavior, or when the host
///   has plenty of RAM and cold-fault latency is not desired.
///
/// - ``enabled(coldFraction:)`` — always instantiate, with a fixed
///   fraction of routed-MoE bytes treated as cold (`0.0 ... 0.95`).
///   `0.70` is the production-recommended value for 128 GB hosts.
///
/// - ``auto(envFallback:)`` — pick at load time:
///   1. If `envFallback == true`, honor `MLXPRESS` env var, with
///      `JANGPRESS` accepted as a legacy fallback:
///      `0`/`off`/`false` → disabled; integer `N` in `[0, 95]` →
///      `.enabled(coldFraction: Double(N) / 100.0)`.
///   2. Otherwise: enable with `coldFraction = 0.70` iff the bundle
///      is detected as routed (MoE) AND raw bytes exceed 50% of
///      physical memory. Otherwise disabled.
public enum JangPressPolicy: Sendable, Equatable {
    case disabled
    case enabled(coldFraction: Double)
    case auto(envFallback: Bool)

    /// Production default — `.auto` with env fallback on. Behavior
    /// adapts per host and respects an explicit `MLXPRESS=` override.
    public static let `default`: JangPressPolicy = .auto(envFallback: true)

    /// Equality treats `coldFraction` bit-exactly (intended; consumers
    /// should pass canonical doubles like 0.70, not arithmetic that
    /// might round).
    public static func == (lhs: JangPressPolicy, rhs: JangPressPolicy) -> Bool {
        switch (lhs, rhs) {
        case (.disabled, .disabled): return true
        case (.enabled(let a), .enabled(let b)): return a == b
        case (.auto(let a), .auto(let b)): return a == b
        default: return false
        }
    }
}

/// Hard cap on resident weight bytes during load. Independent of
/// `MLXPressPolicy` — a caller may disable MLXPress and still want a
/// fail-loud cap, or enable MLXPress without any cap.
///
/// - ``unlimited`` — no cap. Loader behaves as it always has;
///   `MLX_set_wired_limit` and the `MLX.Memory.cacheLimit` are
///   untouched. Current behavior pre-iter-25.
///
/// - ``fraction(_:)`` — cap at `fraction × ProcessInfo.physicalMemory`.
///   `0.70` is the production-recommended value for 128 GB hosts.
///
/// - ``absolute(_:)`` — cap at exactly N bytes. Useful for CI / VMs
///   with restricted memory limits independent of physical RAM.
public enum ResidentCap: Sendable, Equatable {
    case unlimited
    case fraction(Double)
    case absolute(UInt64)

    /// Production default — 70% of physical RAM. Matches the
    /// `MLXPress=70` recommendation but is enforced even when
    /// MLXPress is disabled.
    public static let `default`: ResidentCap = .fraction(0.70)

    /// Resolve to a concrete byte cap given physical RAM. Returns
    /// `nil` for `.unlimited`.
    public func resolve(physicalMemory: UInt64) -> UInt64? {
        switch self {
        case .unlimited:
            return nil
        case .fraction(let f):
            let clamped = max(0.0, min(1.0, f))
            return UInt64(Double(physicalMemory) * clamped)
        case .absolute(let bytes):
            return bytes
        }
    }
}

/// Top-level load-time configuration. Wraps the cold-tier policy and
/// resident cap so osaurus can stash a single Sendable struct in its
/// settings store and pass it through to `loadModel` / `loadContainer`.
///
/// **Source-compat**: this is purely additive. Existing callers that
/// use `loadModel(from:using:)` (no MLXPress) or
/// `loadModel(from:using:jangPress:)` (legacy explicit `JangPressLoadOptions`)
/// keep working unchanged. The new
/// `loadModel(from:using:loadConfiguration:)` overload (added in
/// step 2) consumes this struct.
public struct LoadConfiguration: Sendable, Equatable {
    /// Cold-weight (MLXPress) policy.
    public var jangPress: JangPressPolicy

    /// Hard cap on resident weight bytes (deprecated semantics — use
    /// ``memoryLimit`` instead). Kept for backwards-source-compat.
    /// In the current implementation ``maxResidentBytes`` controls
    /// `MLX.Memory.cacheLimit` only (allocator pool reuse cap), while
    /// ``memoryLimit`` controls the total MLX allocation budget.
    public var maxResidentBytes: ResidentCap

    /// Cap on the total MLX allocation budget. Maps directly onto
    /// `MLX.Memory.memoryLimit`. Calls to malloc will wait on scheduled
    /// tasks if the limit is exceeded — this is how we prevent the
    /// JANGTQ stack-materialization spike from OOM-killing the process
    /// (Ralph iter-14 / Python `_apply_wired_limit_safe_default`
    /// equivalent on Swift).
    ///
    /// Resolved at load entry to `min(fraction × physicalMemory,
    /// recommendedMaxWorkingSetBytes)` so the 847a8c7 crash condition
    /// ("limit larger than max working set size") is impossible by
    /// construction. The clamp matches MLX-swift's
    /// `WiredSumPolicy.clamp(...)` semantics.
    ///
    /// Default `.fraction(0.70)` — Python's tested-good production
    /// value, capped to whichever is smaller of physical RAM and the
    /// GPU's recommended max working set.
    public var memoryLimit: ResidentCap

    /// Use MLX's mmap-backed safetensors loader when the pinned
    /// `mlx-swift` build supports it. For MLXPress `.mmap` loads, the
    /// Osaurus MLX fork maps safetensors tensors as page-aligned
    /// no-copy Metal buffers and creates MLX tensor views into those
    /// buffers, avoiding the stock `pread()` copy into anonymous
    /// allocator storage. Older MLX pins ignore the environment knobs,
    /// so this is source- and runtime-compatible during the pin rollout.
    ///
    /// Default `true` because this is the part that makes cold weights
    /// file-backed and reclaimable under pressure. Set `false` for
    /// strict pre-mmap loader parity.
    public var useMmapSafetensors: Bool

    /// Load preserved native-MTP tensors and instantiate model-owned MTP
    /// modules when the bundle has real tensor evidence and the family is
    /// supported by ``NativeMTPActivation``.
    ///
    /// This is intentionally a load-time switch, separate from per-request
    /// ``GenerateParameters/draftStrategy``. A request can only use native MTP
    /// if the model was loaded with the sidecar weights present; conversely,
    /// loading the sidecar does not force speculative decode for every request.
    public var nativeMTP: Bool

    /// Production default — MLXPress `.disabled` (opt-in), 70% cache +
    /// memory caps, mmap-backed safetensors enabled. Osaurus and other
    /// host integrators get spike-survival memory caps and the patched
    /// mmap loader without engaging the experimental routed-expert
    /// cold-tier. Callers that have validated MLXPress for a specific
    /// bundle family should pass an explicit `.enabled(coldFraction:)`
    /// or `.auto(envFallback:)` policy.
    public static let `default` = LoadConfiguration()

    /// Opt-in MLXPress with auto-detection (`.auto(envFallback: true)`).
    /// Use this when the host has validated MLXPress for the bundles it
    /// serves AND wants the iter-26 routed-MoE cold-tier on bundles
    /// whose total weight bytes exceed half of physical RAM.
    /// `MLXPRESS=N` (0–95) and `MLXPRESS=off` env overrides honored;
    /// legacy `JANGPRESS=*` remains accepted.
    public static let experimentalJangPressAuto = LoadConfiguration(
        jangPress: .auto(envFallback: true),
        maxResidentBytes: .default,
        memoryLimit: .default,
        useMmapSafetensors: true,
        nativeMTP: false)

    /// Production host preset for Osaurus/vMLX server integrations.
    ///
    /// This keeps the conservative load-time memory caps and mmap
    /// safetensors path from ``default``, while opting validated hosts
    /// into MLXPress auto-detection for routed bundles that exceed half
    /// of physical RAM. Native MTP remains a separate per-bundle
    /// decision resolved by ``VMLXServerRuntimeSettings``.
    public static let osaurusProduction = experimentalJangPressAuto

    /// Everything off — strict byte-compat with pre-iter-23 behavior.
    /// MLXPress disabled, no caps, no mmap loader.
    public static let off = LoadConfiguration(
        jangPress: .disabled,
        maxResidentBytes: .unlimited,
        memoryLimit: .unlimited,
        useMmapSafetensors: false,
        nativeMTP: false)

    public init(
        jangPress: JangPressPolicy = .disabled,
        maxResidentBytes: ResidentCap = .default,
        memoryLimit: ResidentCap = .default,
        useMmapSafetensors: Bool = true,
        nativeMTP: Bool = false
    ) {
        self.jangPress = jangPress
        self.maxResidentBytes = maxResidentBytes
        self.memoryLimit = memoryLimit
        self.useMmapSafetensors = useMmapSafetensors
        self.nativeMTP = nativeMTP
    }
}

// MARK: - Resolution

/// Snapshot of the bundle facts needed by `MLXPressPolicy.resolve` —
/// pulled once at load entry so the resolver doesn't re-walk the
/// directory or re-parse `config.json`.
public struct LoadBundleFacts: Sendable, Equatable {
    /// Sum of `*.safetensors` byte sizes in the bundle. `0` when
    /// inspection failed (treat as unknown — `.auto` falls through to
    /// disabled in that case).
    public var totalSafetensorsBytes: UInt64

    /// True iff `config.json` declares any of:
    /// `num_local_experts`, `num_experts`, `moe_intermediate_size`,
    /// `n_routed_experts`, or contains a non-empty `experts` mapping.
    /// `false` when inspection failed (treat as dense).
    public var isRouted: Bool

    /// Best-effort routed-expert count sniffed from `config.json`.
    /// Used by router-aware MLXPress advice to size each layer's hot
    /// set. `nil` when the bundle does not expose the shape.
    public var numRoutedExperts: Int?

    /// Best-effort top-k routed experts per token sniffed from
    /// `config.json`. `nil` falls back to the advisor's conservative
    /// default.
    public var topK: Int?

    /// Physical memory snapshot captured at the same moment, so cap
    /// math stays internally consistent with the bundle inspection.
    public var physicalMemory: UInt64

    public init(
        totalSafetensorsBytes: UInt64,
        isRouted: Bool,
        physicalMemory: UInt64,
        numRoutedExperts: Int? = nil,
        topK: Int? = nil
    ) {
        self.totalSafetensorsBytes = totalSafetensorsBytes
        self.isRouted = isRouted
        self.physicalMemory = physicalMemory
        self.numRoutedExperts = numRoutedExperts
        self.topK = topK
    }

    /// Inspect the bundle directory at `url`. All probes are best-effort —
    /// failures degrade silently to safe defaults (zero bytes, dense).
    public static func inspect(bundleURL url: URL) -> LoadBundleFacts {
        let fm = FileManager.default
        let physical = ProcessInfo.processInfo.physicalMemory

        // Walk top-level for *.safetensors. Don't recurse — MLXPress
        // bundles never nest weights under subdirs.
        var totalBytes: UInt64 = 0
        if let entries = try? fm.contentsOfDirectory(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles])
        {
            for entry in entries where entry.pathExtension == "safetensors" {
                if let values = try? entry.resourceValues(forKeys: [.fileSizeKey]),
                    let size = values.fileSize
                {
                    totalBytes &+= UInt64(size)
                }
            }
        }

        // Probe config.json for routed markers.
        var routed = false
        var numRoutedExperts: Int?
        var topK: Int?
        let configURL = url.appendingPathComponent("config.json")
        if let data = try? Data(contentsOf: configURL),
            let json = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any]
        {
            let routedKeys = [
                "num_local_experts",
                "num_experts",
                "moe_intermediate_size",
                "n_routed_experts",
            ]
            let routedExpertKeys = [
                "n_routed_experts",
                "num_local_experts",
                "num_experts",
            ]
            let topKKeys = [
                "num_experts_per_tok",
                "top_k_experts",
                "experts_per_token",
            ]
            func firstPositiveInt(
                in object: [String: Any],
                keys: [String]
            ) -> Int? {
                for key in keys {
                    if let n = object[key] as? Int, n > 0 {
                        return n
                    }
                    if let n = object[key] as? NSNumber, n.intValue > 0 {
                        return n.intValue
                    }
                }
                return nil
            }

            numRoutedExperts = firstPositiveInt(
                in: json, keys: routedExpertKeys)
            topK = firstPositiveInt(in: json, keys: topKKeys)
            for key in routedKeys {
                if let n = json[key] as? Int, n > 1 {
                    routed = true
                    break
                }
            }
            // Some VLMs nest the LM stanza under `text_config` /
            // `language_config`; check those too. Even when the
            // top-level wrapper already proved "routed", nested
            // values may be the only source for `num_experts_per_tok`.
            for sub in ["text_config", "language_config"] {
                if let nested = json[sub] as? [String: Any] {
                    if numRoutedExperts == nil {
                        numRoutedExperts = firstPositiveInt(
                            in: nested, keys: routedExpertKeys)
                    }
                    if topK == nil {
                        topK = firstPositiveInt(in: nested, keys: topKKeys)
                    }
                    if !routed {
                        for key in routedKeys {
                            if let n = nested[key] as? Int, n > 1 {
                                routed = true
                                break
                            }
                        }
                    }
                }
            }
            if (numRoutedExperts ?? 0) > 1 {
                routed = true
            }
        }

        return LoadBundleFacts(
            totalSafetensorsBytes: totalBytes,
            isRouted: routed,
            physicalMemory: physical,
            numRoutedExperts: numRoutedExperts,
            topK: topK)
    }
}

extension JangPressPolicy {
    /// Resolve to a concrete `MLXPressLoadOptions` for the given
    /// bundle. Precedence: explicit value > env (`MLXPRESS=...`, then
    /// legacy `JANGPRESS=...`) >
    /// `.auto` threshold (routed AND bundleBytes > 0.5 × physicalMemory).
    ///
    /// The default cold fraction for `.auto` is `0.70` (matches the
    /// production tuning verified in iter 25 across 4 model families).
    public func resolve(facts: LoadBundleFacts) -> JangPressLoadOptions {
        switch self {
        case .disabled:
            return .disabled

        case .enabled(let coldFraction):
            let pct = Int((max(0.0, min(0.95, coldFraction)) * 100.0).rounded())
            return JangPressLoadOptions(enabled: true, compressPct: pct)

        case .auto(let envFallback):
            if envFallback,
                let raw = ProcessInfo.processInfo.environment["MLXPRESS"]
                    ?? ProcessInfo.processInfo.environment["JANGPRESS"]
            {
                let lowered = raw.lowercased()
                if lowered == "0" || lowered == "off" || lowered == "false" {
                    return .disabled
                }
                if let n = Int(raw), (0...95).contains(n) {
                    return JangPressLoadOptions(enabled: true, compressPct: n)
                }
                // Garbage env value — fall through to threshold.
            }

            // Threshold rule: enable iff routed AND bundle bytes
            // exceed half of physical RAM. Conservative on dense
            // models (no benefit) and tiny MoE (no pressure).
            let halfRAM = facts.physicalMemory / 2
            if facts.isRouted && facts.totalSafetensorsBytes > halfRAM {
                return JangPressLoadOptions(enabled: true, compressPct: 70)
            }
            return .disabled
        }
    }
}

// MARK: - Inspection

/// Runtime status of the MLXPress runtime — used by osaurus / JANG
/// Studio settings panels. **Drastically simplified in iter 26**: the
/// controller state machine and routing counters were removed because
/// nothing in production drove them. What remains is the mmap probe's
/// tile-classification snapshot (used for UI bookkeeping like "how
/// many routed-expert tiles did we identify in this bundle?").
///
/// For real memory metrics (heap budget, RSS, recommended working set)
/// use ``MemoryStatus/snapshot()`` — that's the surface that reflects
/// the actual cap that ships today via `LoadConfiguration.memoryLimit`.
public struct JangPressStatus: Sendable, Equatable {
    /// True iff at least one tier (mmap probe or embed cache) is
    /// attached. Purely informational — nothing about cold-tier
    /// reclaim or controller state is implied by this flag today.
    public let enabled: Bool

    /// Cold-fraction setting captured at activation time, expressed as
    /// `0.0 ... 0.95`. Today this controls how aggressively the embed
    /// tier shrinks its hot row count; the routed-expert tier no
    /// longer drives any madvise calls. `nil` when disabled.
    public let coldFraction: Double?

    /// Routed-expert backend chosen at activation time. With `.mach`
    /// removed in iter 26 the only meaningful values are `.mmap`
    /// (probe attached) or `.none`.
    public let backend: JangPressLoadOptions.Backend

    /// Number of routed-expert tiles classified by the mmap probe.
    /// `0` when no mmap probe is attached or when the lazy probe has
    /// not yet been built (`snapshotIfBuilt()` returns zeros until
    /// something forces it).
    public let tilesUnderManagement: Int

    /// Total bytes of routed-expert weights identified by the mmap
    /// probe. `0` when the probe is not built / not attached.
    public let totalRoutedBytes: UInt64

    /// Empty-status sentinel — represents `JangPressRuntime.none`.
    public static let disabled = JangPressStatus(
        enabled: false,
        coldFraction: nil,
        backend: .none,
        tilesUnderManagement: 0,
        totalRoutedBytes: 0)

    public init(
        enabled: Bool,
        coldFraction: Double?,
        backend: JangPressLoadOptions.Backend,
        tilesUnderManagement: Int,
        totalRoutedBytes: UInt64
    ) {
        self.enabled = enabled
        self.coldFraction = coldFraction
        self.backend = backend
        self.tilesUnderManagement = tilesUnderManagement
        self.totalRoutedBytes = totalRoutedBytes
    }
}

extension JangPressRuntime {
    /// Snapshot the current state of the JangPress probes. Returns
    /// ``JangPressStatus/disabled`` for ``JangPressRuntime/none``.
    public func status() -> JangPressStatus {
        guard isActive else { return .disabled }

        let backend: JangPressLoadOptions.Backend
        var totalRoutedBytes: UInt64 = 0
        var tiles: Int = 0
        if let mmap {
            backend = .mmap
            let s = mmap.snapshotIfBuilt()
            totalRoutedBytes = s.totalRoutedBytes
            tiles = s.expertCount
        } else {
            backend = .none
        }

        // Cold fraction is recovered from the appliedOptions stash
        // (set by `JangPressActivation.activate`). compressPct is the
        // % of routed mass open to compression; coldFraction is the
        // same value as a 0..1 double for consumer ergonomics.
        let coldFraction: Double? = appliedOptions.map {
            Double($0.compressPct) / 100.0
        }

        return JangPressStatus(
            enabled: true,
            coldFraction: coldFraction,
            backend: backend,
            tilesUnderManagement: tiles,
            totalRoutedBytes: totalRoutedBytes)
    }
}

extension ResidentCap {
    /// Apply the cap to MLX's allocator cache for the duration of a
    /// scoped block (typically one load call). Returns the prior limit
    /// so the caller can restore it via `defer`. `nil` return means
    /// `.unlimited` — caller should leave the limit untouched.
    ///
    /// Note: `MLX.Memory.cacheLimit` is the allocator's internal pool,
    /// not a hard RSS ceiling. Capping it forces freed intermediates
    /// to release back to the OS instead of accumulating in the pool —
    /// the same mechanism iter 25 uses with its hard-coded 1 GB.
    public func applyAsCacheLimitInt(physicalMemory: UInt64) -> Int? {
        guard let bytes = self.resolve(physicalMemory: physicalMemory) else {
            return nil
        }
        // Clamp to Int.max — `MLX.Memory.cacheLimit` is `Int`. On
        // 64-bit platforms this is 9.2 EB so the clamp is a formality.
        return Int(min(UInt64(Int.max), bytes))
    }
}

// MARK: - MemoryStatus

/// Snapshot of MLX memory caps + current process RSS. Designed for UI
/// display (osaurus settings panel, JANG Studio inspector). All fields
/// are plain values; cheap enough to poll on a timer.
///
/// Pull via ``MemoryStatus/snapshot()`` — bypasses any container plumbing
/// because all four MLX limits are process-global.
public struct MemoryStatus: Sendable, Equatable {
    /// Total MLX allocation budget. Calls to malloc wait if exceeded.
    /// Set at load entry to `min(LoadConfiguration.memoryLimit fraction
    /// × physicalMemory, recommendedMaxWorkingSetBytes)`.
    public let memoryLimit: Int

    /// Allocator pool reuse cap. Freed intermediates above this size
    /// release back to the OS instead of accumulating in the cache.
    public let cacheLimit: Int

    /// MTLDevice's recommended max working set size — the upper bound
    /// for any wired-limit setting (exceeding it triggers the 847a8c7
    /// crash). nil when Metal is unavailable.
    public let recommendedWorkingSetBytes: Int?

    /// System physical RAM as reported by `sysctl(HW_MEMSIZE)`.
    public let physicalMemory: UInt64

    /// Current process resident set size sampled from
    /// `mach_task_basic_info`. Reflects ALL process memory (MLX +
    /// frameworks + stack + heap), not just MLX-allocated tensors.
    public let currentRSS: UInt64

    /// Capture all current MLX caps + a fresh RSS sample.
    public static func snapshot() -> MemoryStatus {
        MemoryStatus(
            memoryLimit: MLX.Memory.memoryLimit,
            cacheLimit: MLX.Memory.cacheLimit,
            recommendedWorkingSetBytes: MLX.GPU.maxRecommendedWorkingSetBytes(),
            physicalMemory: ProcessInfo.processInfo.physicalMemory,
            currentRSS: residentSetSizeBytes())
    }
}

#if canImport(Darwin)
import Darwin

/// Sample the current process RSS via mach_task_basic_info. Returns 0
/// on failure (rare — the syscall is well-behaved on macOS / iOS).
private func residentSetSizeBytes() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(
        MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_,
                task_flavor_t(MACH_TASK_BASIC_INFO),
                $0, &count)
        }
    }
    guard kr == KERN_SUCCESS else { return 0 }
    return UInt64(info.resident_size)
}
#else
private func residentSetSizeBytes() -> UInt64 { 0 }
#endif
