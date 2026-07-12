// The prefix cache is an optimisation: a stored entry only ever makes some later
// request faster. Storing one must therefore never be able to take the host down.
//
// It could. `storeCacheAfterGeneration` materialises the KV cache several times
// over — a deep copy (`Evaluate.swift`), a host `Data` extract for the disk write,
// and (when quantizing) another copy in `makeDiskStoreCache` — at the moment memory
// is already at its high-water mark. Reported live on an M5 Max / 128 GB running
// Llama-3.3-70B-8bit at a 64K context: the reporter's own footprint time series
// showed prefill tracking expectation (75 -> 102 GiB), then +23 GiB immediately
// after generation, inside the cache-store window — ~23 GiB being exactly the size
// of the 62K-token KV cache. Free memory collapsed, page reclaim stalled, and the
// kernel panicked. macOS will not jetsam a plain user process out of the way, so
// nothing intervenes: the cap has to live here.
//
// These tests pin the arithmetic that decides whether the copies fit.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("Cache-store memory budget")
struct CacheStoreBudgetTests {

    private static let gib = 1 << 30

    /// The reported panic, in numbers: a 70B 8-bit model already holds ~70 GB of
    /// weights, the 64K-token KV cache is ~20 GiB more, and the store then wants
    /// several more copies of that cache. It does not fit, and must be refused.
    @Test("The 70B-at-64K store that panicked a 128 GB host is refused")
    func longContextStoreOnFullHostIsRefused() {
        let kv = 20 * Self.gib  // 80 layers x 2 x 8 kv-heads x 128 dim x 65536 x fp16
        let active = 92 * Self.gib  // 70 GB weights + the live KV, already resident
        let budget = 128 * Self.gib
        #expect(
            !CacheStoreBudget.canStore(cacheBytes: kv, activeBytes: active, budgetBytes: budget),
            "storing this cache needs tens of GiB the host does not have — it must be skipped"
        )
    }

    /// The guard must not disable the prefix cache for ordinary work. A normal chat
    /// turn's cache is a rounding error against the budget.
    @Test("An ordinary cache on a healthy host still stores")
    func ordinaryStoreIsAllowed() {
        let kv = 512 * (1 << 20)  // 512 MiB — a few thousand tokens
        let active = 40 * Self.gib
        let budget = 128 * Self.gib
        #expect(
            CacheStoreBudget.canStore(cacheBytes: kv, activeBytes: active, budgetBytes: budget))
    }

    /// The same cache that is refused on a full host is fine on an empty one:
    /// the verdict is about headroom, not about the cache being "too big".
    @Test("Headroom, not size, decides")
    func sameCacheFitsWhenTheHostIsEmpty() {
        let kv = 20 * Self.gib
        let budget = 128 * Self.gib
        #expect(
            !CacheStoreBudget.canStore(
                cacheBytes: kv, activeBytes: 92 * Self.gib, budgetBytes: budget))
        #expect(
            CacheStoreBudget.canStore(
                cacheBytes: kv, activeBytes: 8 * Self.gib, budgetBytes: budget))
    }

    /// The refusal has to fire *before* the allocations, so the margin covers the
    /// copies the store is about to make — not just the one it already made.
    @Test("The budget counts every copy the store will make, plus a safety margin")
    func budgetCountsAllMaterializations() {
        #expect(CacheStoreBudget.materializationFactor >= 2)
        #expect(CacheStoreBudget.safetyMarginBytes > 0)

        // A cache sized so that one copy fits but the full store does not: the
        // guard must refuse it. This is the case that panicked the host — the
        // first copy succeeds, and the machine dies on a later one.
        let budget = 128 * Self.gib
        let active = 100 * Self.gib
        let headroom = budget - active  // 28 GiB
        let kv = headroom / 2  // one copy fits; factor x copies do not
        #expect(
            !CacheStoreBudget.canStore(cacheBytes: kv, activeBytes: active, budgetBytes: budget))
    }

    /// The guard must not quietly disable the prefix cache for the long-context
    /// requests that most need it. With a 94 GiB pack resident on a 128 GB Mac, a
    /// multi-thousand-token conversation still caches — it is only the runaway case
    /// that is refused. (Budgeting against the GPU working set instead of physical
    /// memory would cut this off around 15k tokens, which is why it doesn't.)
    @Test("A long conversation still caches with a large model resident")
    func longContextStillCachesUnderALargeResidentModel() {
        let physical = 128 * Self.gib
        let active = 96 * Self.gib  // a 94 GiB pack, resident
        // Hy3 KV: 80 layers x 2 x 8 kv-heads x 128 dim x fp16 = 320 KiB/token.
        let bytesPerToken = 80 * 2 * 8 * 128 * 2
        for tokens in [2_000, 8_000, 16_000] {
            #expect(
                CacheStoreBudget.canStore(
                    cacheBytes: tokens * bytesPerToken,
                    activeBytes: active,
                    budgetBytes: physical),
                "a \(tokens)-token cache must still be storable on a healthy host"
            )
        }
    }

    /// No budget information is not a licence to block: an unknown budget means we
    /// cannot reason, and a prefix cache that silently stops working everywhere
    /// would be its own bug.
    @Test("An unknown budget does not disable the cache")
    func unknownBudgetDoesNotBlock() {
        #expect(CacheStoreBudget.canStore(cacheBytes: 1 << 30, activeBytes: 0, budgetBytes: nil))
        #expect(CacheStoreBudget.canStore(cacheBytes: 0, activeBytes: 0, budgetBytes: 0))
    }
}
