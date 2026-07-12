// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Whether a KV cache can be *saved* without pushing the host over a cliff.
///
/// The prefix cache is an optimisation: a stored entry only ever makes a later
/// request faster. Storing one must therefore never be able to take the machine
/// down — but until this guard existed, it could.
///
/// `storeCacheAfterGeneration` materialises the cache up to three times over,
/// at the exact moment memory is already at its high-water mark:
///
///   1. `cacheToStore.map { $0.copy() }`   — a full duplicate of the live KV
///   2. `extractLayerData(from: snapshot)` — again, as host `Data`, for the disk write
///   3. `makeDiskStoreCache(...)`          — and again, as the disk-store cache
///
/// For a 70B 8-bit model with a 64K-token context that is ~70 GB of weights plus
/// a ~20 GiB live KV cache, and then the store adds tens of GiB more. On a 128 GB
/// host, free memory collapses. macOS will not jetsam a plain user process out of
/// the way, so nothing intervenes: page reclaim stalls, watchdogd starves, and the
/// kernel panics — reported live on an M5 Max/128 GB with Llama-3.3-70B-8bit at 64K,
/// with the panic landing inside `storeCacheAfterGeneration` / `DiskCache.store`.
///
/// So: measure first, and if the copies would not fit, skip the store. A skipped
/// store costs one slower request. An unchecked store costs the machine.
public enum CacheStoreBudget {

    /// How many times the store materialises the cache (snapshot + host-`Data`
    /// extract + disk-store cache). Deliberately conservative: undercounting here
    /// is what made the original panic possible.
    static let materializationFactor = 3

    /// Headroom kept free for everything else in flight — activations, the disk
    /// write buffer, and the rest of the system.
    ///
    /// Scaled to the host, not fixed. A flat 4 GiB reserve is right on a 128 GB
    /// Mac and nonsense on an 8 GB one: with a 4.5 GB model resident, `active +
    /// 0 + 4 GiB` already exceeds 8 GB, so the guard would refuse EVERY store —
    /// silently disabling the prefix cache on exactly the machines whose users
    /// most notice a slow re-prefill. An eighth of RAM keeps the same absolute
    /// headroom on large hosts (128 GB → 4 GiB, unchanged) while staying
    /// proportionate on small ones (8 GB → 1 GiB).
    static func safetyMarginBytes(budgetBytes: Int) -> Int {
        min(4 << 30, budgetBytes / 8)
    }

    /// Live bytes held by a KV cache, without copying its contents.
    ///
    /// `state` is the same array set the store would serialise, and `nbytes` is
    /// shape x item size, so nothing is evaluated and no tensor data is copied.
    /// Some implementations do build lazy views to answer it (slices in
    /// `KVCacheSimple` / `RotatingKVCache` / `QuantizedKVCache`, a flatten in
    /// `CacheList`), so this is cheap rather than literally free — orders of
    /// magnitude below the copy + serialize it is deciding whether to allow.
    public static func cacheBytes(_ cache: [KVCache]) -> Int {
        cache.reduce(0) { total, layer in
            total + layer.state.reduce(0) { $0 + $1.nbytes }
        }
    }

    /// Whether storing `cache` fits in what the host can still give us.
    ///
    /// Budgeted against **physical memory**, not the GPU working set. The working
    /// set is the wrong ceiling for this decision in both directions: exceeding it
    /// only means macOS pages the excess — slow, survivable — whereas exhausting
    /// physical memory is what actually kills the host, and that is the only thing
    /// this guard exists to prevent.
    ///
    /// Using the working set here would also be needlessly destructive: with a
    /// large model resident (say 96 GB of a 107 GiB working set) it would refuse to
    /// cache anything past ~15k tokens, disabling the prefix cache for exactly the
    /// long-context requests that most need it. Against physical memory the same
    /// host still caches ~38k tokens, and the request that panicked a 128 GB Mac is
    /// still refused.
    public static func canStore(_ cache: [KVCache]) -> Bool {
        let liveBytes = cacheBytes(cache)
        guard liveBytes > 0 else { return true }
        return canStore(cacheBytes: liveBytes)
    }

    /// Testable core: no MLX state, just the arithmetic.
    static func canStore(
        cacheBytes liveBytes: Int,
        activeBytes: Int = max(0, MLX.Memory.activeMemory),
        budgetBytes: Int? = Int(exactly: ProcessInfo.processInfo.physicalMemory)
    ) -> Bool {
        guard liveBytes > 0 else { return true }
        guard let budgetBytes, budgetBytes > 0 else {
            // No budget to reason about: don't guess, don't block.
            return true
        }
        let storeCost = liveBytes * materializationFactor
        let projected = activeBytes + storeCost + safetyMarginBytes(budgetBytes: budgetBytes)
        return projected <= budgetBytes
    }
}
