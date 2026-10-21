// Copyright © 2026 Jinho Jang. All rights reserved.
//
// JangPressActivation — instantiate the JangPress status/probe tiers
// (mmap probe + embed cache).
//
// **Post-2026-05-03 cleanup**: the controller + .mach backend were
// removed because they advised a parallel mapping rather than MLX's
// canonical tensor storage. This activator now builds the mmap
// tile-classification probe and the orthogonal Zipfian embed cache.
// Router-aware canonical page advice has been removed.

import Foundation

public enum JangPressActivation {

    /// Instantiate the surviving JangPress tiers per `options`. Returns
    /// `JangPressRuntime.none` when `options.enabled == false` (the
    /// default) so callers can wire this in unconditionally.
    ///
    /// Failure modes are non-fatal: any thrown error from a tier's
    /// constructor is logged to stderr and that tier returns nil. The
    /// runtime is still returned so the caller's inference path keeps
    /// working with full-resident weights.
    ///
    /// - Parameters:
    ///   - bundleURL: the model bundle directory.
    ///   - options: `JangPressLoadOptions(enabled: ..., compressPct:
    ///     ..., backend: ...)`. The `forceMode` and `enablePrefetch`
    ///     fields exist on `JangPressLoadOptions`; `enablePrefetch`
    ///     affects the probe tier's start-cold behavior. Router advice is retired.
    /// - Returns: `JangPressRuntime` with `mmap` and/or `embed`
    ///   populated per the options.
    public static func activate(
        bundleURL: URL,
        options: JangPressLoadOptions
    ) -> JangPressRuntime {
        guard options.enabled else { return .none }

        // Routed-expert mmap PROBE — kept for tile classification and
        // UI status. The actual RSS savings come from the patched MLX
        // safetensors loader's canonical mmap-backed arrays.
        var mmapTier: JangPressMmapTier?
        switch options.backend {
        case .mmap:
            do {
                let cfg = JangPressMmapConfig(
                    bundleURL: bundleURL,
                    hotPercent: 100 - options.compressPct,
                    startCold: !options.enablePrefetch)
                mmapTier = try JangPressMmapTier(config: cfg)
            } catch {
                FileHandle.standardError.write(Data(
                    "[MLXPressActivation] mmap-tier init failed: \(error)\n".utf8))
            }
        case .none:
            break
        }

        // Build the mmap probe immediately after model load so status
        // surfaces report real routed tile counts on the first poll.
        // This parses safetensors headers and mmaps routed shards, but
        // it does not copy tensor bytes or enter the decode hot path.
        if let mmapTier {
            _ = mmapTier.snapshot()
        }

        // Embed/lm_head Zipfian tier — orthogonal cache, still useful
        // independently of the routed-expert path.
        var embedTier: JangPressEmbedTier?
        do {
            let embedHot = max(1, min(50, 30 - (options.compressPct / 4)))
            let cfg = JangPressEmbedConfig(
                bundleURL: bundleURL,
                hotPercent: embedHot,
                skipLMHead: false)
            embedTier = try JangPressEmbedTier(config: cfg)
        } catch {
            FileHandle.standardError.write(Data(
                "[MLXPressActivation] embed-tier init failed (non-fatal): \(error)\n".utf8))
        }

        return JangPressRuntime(
            mmap: mmapTier, embed: embedTier, appliedOptions: options)
    }

    /// Drop a runtime. Currently a no-op kept for source-compat with
    /// callers that wrap activate/deactivate in defer blocks. The
    /// tiers release on the runtime's last reference drop (Swift ARC).
    public static func deactivate(_ runtime: JangPressRuntime) {
        _ = runtime
    }
}
