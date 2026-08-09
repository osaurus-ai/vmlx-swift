// Copyright © 2026 Apple Inc.

import Foundation

/// A cache layer whose architecture retains live Metal buffers for every
/// decoded token.
///
/// Metal caps the NUMBER of live MLX buffers, not just their bytes: on an
/// M5 Max the ceiling (`iogpu.rsrc_limit`) is 499000, and exceeding it raises
/// `[metal::malloc] Resource limit (499000) exceeded` mid-generation.
/// Conventional KV caches grow in steps and reuse buffers, so their live
/// count stays flat and they must never report a rate here — a family whose
/// retention has not been measured must not be capped on a limit it provably
/// cannot reach. DSV4's cumulative compressor/indexer pools are the measured
/// exception (~one buffer per layer per decode token), and `clear_cache`
/// frees none of that retained state, so the ceiling has to be projected
/// rather than recovered from.
public protocol MetalBufferRetainingCache {
    /// Live MLX buffers this cache layer retains per decoded token.
    var retainedBuffersPerDecodeToken: Double { get }
}

/// Projects the Metal live-buffer ceiling into a safe generated-token cap and
/// clamps a request's `maxTokens` beneath it. Inert (returns the requested
/// value unchanged) for cache topologies that retain no per-token buffers.
///
/// Env knobs (parity with Python vMLX 1.6.24 / R21-13):
/// - `VMLX_METAL_BUFFER_COUNT_GUARD` — `0` disables the guard (default on).
/// - `VMLX_METAL_BUFFER_GUARD_FRACTION` — safety fraction of the resource
///   limit to budget (default 0.90).
/// - `VMLX_METAL_BUFFER_GUARD_BASELINE` — live buffers reserved for model
///   weights plus prompt-side state before decoding starts (default 16384).
public enum MetalLiveBufferGuard {

    public static let defaultSafetyFraction: Double = 0.90
    public static let defaultBaselineBuffers: Int = 16384

    /// Live-buffer ceiling reported by the OS, or `nil` when unavailable.
    ///
    /// Mirrors mlx's Metal `device_info()`: reads `iogpu.rsrc_limit` and
    /// falls back to 499000 when the sysctl reports nothing.
    public static func resourceLimit() -> Int? {
        var value: Int = 0
        var length = MemoryLayout<Int>.size
        sysctlbyname("iogpu.rsrc_limit", &value, &length, nil, 0)
        if value <= 0 { value = 499_000 }
        return value
    }

    /// Safe generated-token cap from the Metal live-buffer ceiling.
    ///
    /// Returns `nil` when the architecture retains no per-token buffers or
    /// the ceiling is unknown; returns `0` when the baseline alone exhausts
    /// the budget.
    public static func projectedOutputTokenCap(
        resourceLimit: Int,
        liveBaselineBuffers: Int,
        buffersPerToken: Double,
        safetyFraction: Double = MetalLiveBufferGuard.defaultSafetyFraction
    ) -> Int? {
        guard resourceLimit > 0, buffersPerToken > 0 else { return nil }
        var fraction = safetyFraction
        if !(fraction > 0 && fraction <= 1) {
            fraction = defaultSafetyFraction
        }
        let budget = Int(Double(resourceLimit) * fraction) - max(0, liveBaselineBuffers)
        if budget <= 0 { return 0 }
        return max(0, Int(Double(budget) / buffersPerToken))
    }

    /// Total live buffers retained per decoded token across a cache topology.
    public static func retainedBuffersPerDecodeToken(cache: [KVCache]) -> Double {
        cache.reduce(0.0) {
            $0 + (($1 as? MetalBufferRetainingCache)?.retainedBuffersPerDecodeToken ?? 0.0)
        }
    }

    /// Clamps a request's `maxTokens` to the projected buffer-count cap.
    ///
    /// An unlimited request (`nil`) against a retaining topology is clamped to
    /// the cap — that is exactly the long-generation case that used to die at
    /// the allocator wall. Non-retaining topologies pass through untouched.
    public static func clampedMaxTokens(
        requested: Int?,
        cache: [KVCache],
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int? {
        guard environment["VMLX_METAL_BUFFER_COUNT_GUARD"] != "0" else { return requested }
        let perToken = retainedBuffersPerDecodeToken(cache: cache)
        guard perToken > 0, let limit = resourceLimit() else { return requested }
        let fraction =
            environment["VMLX_METAL_BUFFER_GUARD_FRACTION"].flatMap(Double.init)
            ?? defaultSafetyFraction
        let baseline =
            environment["VMLX_METAL_BUFFER_GUARD_BASELINE"].flatMap(Int.init)
            ?? defaultBaselineBuffers
        guard
            let cap = projectedOutputTokenCap(
                resourceLimit: limit,
                liveBaselineBuffers: baseline,
                buffersPerToken: perToken,
                safetyFraction: fraction)
        else { return requested }
        if let requested, requested <= cap { return requested }
        print(
            "[MetalLiveBufferGuard] Metal live-buffer ceiling is the binding output limit: "
                + "cap=\(cap) tokens (resource_limit=\(limit), "
                + "\(perToken) buffers/token, requested=\(requested.map(String.init) ?? "unlimited"))"
        )
        return cap
    }
}
