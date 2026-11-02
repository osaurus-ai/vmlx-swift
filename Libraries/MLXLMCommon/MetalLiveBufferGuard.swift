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
/// frees none of that retained state. Such measurements can inform diagnostics;
/// they cannot override caller or model output limits.
public protocol MetalBufferRetainingCache {
    /// Live MLX buffers this cache layer retains per decoded token.
    var retainedBuffersPerDecodeToken: Double { get }
}

/// Reports projected live-buffer pressure without changing a request's output limit.
/// An estimate cannot replace the model or caller's generation contract. Resource
/// exhaustion must surface as the actual runtime error, not a fabricated length stop.
///
/// Env knobs (parity with Python vMLX 1.6.24 / R21-13):
/// - `VMLX_METAL_BUFFER_COUNT_GUARD` — `0` disables reporting (default on).
/// - `VMLX_METAL_BUFFER_GUARD_FRACTION` — safety fraction of the resource
///   limit used by the diagnostic projection (default 0.90).
/// - `VMLX_METAL_BUFFER_GUARD_BASELINE` — live buffers reserved for model
///   weights plus prompt-side state before decoding starts (default 16384).
public enum MetalLiveBufferGuard {

    public static let defaultSafetyFraction: Double = 0.90
    public static let defaultBaselineBuffers: Int = 16384

    /// Live-buffer ceiling reported by the OS, or `nil` when unavailable.
    ///
    /// Reads `iogpu.rsrc_limit` when the OS exposes it. An unavailable value
    /// stays unknown; a historical limit from another Mac is not evidence.
    public static func resourceLimit() -> Int? {
        #if canImport(Darwin)
            var value: Int = 0
            var length = MemoryLayout<Int>.size
            guard sysctlbyname("iogpu.rsrc_limit", &value, &length, nil, 0) == 0,
                value > 0
            else { return nil }
            return value
        #else
            return nil
        #endif
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

    /// Report a projection only when the OS limit and cache retention are known.
    /// This never changes the caller's requested output-token limit.
    public static func reportProjectedBufferPressure(
        requested: Int?,
        cache: [KVCache],
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) {
        guard environment["VMLX_METAL_BUFFER_COUNT_GUARD"] != "0" else { return }
        let perToken = retainedBuffersPerDecodeToken(cache: cache)
        guard perToken > 0, let limit = resourceLimit() else { return }
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
        else { return }
        if let requested, requested <= cap { return }
        print(
            "[MetalLiveBufferGuard] Advisory only; requested output limit is unchanged: "
                + "cap=\(cap) tokens (resource_limit=\(limit), "
                + "\(perToken) buffers/token, requested=\(requested.map(String.init) ?? "unlimited"))"
        )
    }

    /// Historical API retained for source compatibility. Buffer projections are
    /// diagnostic only; explicit and omitted output limits pass through unchanged.
    public static func clampedMaxTokens(
        requested: Int?,
        cache: [KVCache],
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int? {
        reportProjectedBufferPressure(requested: requested, cache: cache, environment: environment)
        return requested
    }
}
