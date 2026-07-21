// The prefix cache is an optimisation: a stored entry only ever makes some later
// request faster. Storing one must therefore never be able to take the host down.
//
// It could. Reported live on an M5 Max / 128 GB running Llama-3.3-70B-8bit at a
// 64K context: the reporter's own footprint time series showed prefill tracking
// expectation (75 -> 102 GiB), then +23 GiB immediately after generation, inside
// the cache-store window — ~23 GiB being exactly the size of the 62K-token KV
// cache. Free memory collapsed, page reclaim stalled, and the kernel panicked.
// macOS will not jetsam a plain user process out of the way, so nothing
// intervenes: the cap has to live here.
//
// These tests pin the arithmetic that decides whether the store fits, and pin it
// to the user's memory-safety level rather than to a number we picked.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("Cache-store memory budget")
struct CacheStoreBudgetTests {

    private static let gib = 1 << 30

    /// Every case below runs at the shipped default unless it says otherwise, so
    /// the default is what the assertions are really about.
    private static let safeAuto = CacheStorePolicy.safeAuto

    /// MLX allocator accounting can describe the logical/mapped model while
    /// Activity Monitor shows the much smaller set of pages actually charged to
    /// the process. RAM safety must use the latter or it rejects harmless SSD
    /// checkpoints for reclaimed JANG models.
    @Test("Physical footprint wins over inflated MLX allocator accounting")
    func physicalFootprintWinsOverMLXAccounting() {
        let active = CacheStoreBudget.activeBytesForStoreBudget(
            physicalFootprintBytes: UInt64(19 * Self.gib),
            mlxActiveBytes: 96 * Self.gib
        )
        #expect(active == 19 * Self.gib)
    }

    @Test("MLX active memory remains the fallback when footprint is unavailable")
    func mlxAccountingIsThePortableFallback() {
        let active = CacheStoreBudget.activeBytesForStoreBudget(
            physicalFootprintBytes: nil,
            mlxActiveBytes: 37 * Self.gib
        )
        #expect(active == 37 * Self.gib)
    }

    /// The reported panic, in numbers: a 70B 8-bit model already holds ~70 GB of
    /// weights, the 64K-token KV cache is ~20 GiB more, and the store then wants
    /// another copy of that cache. It does not fit, and must be refused.
    @Test("The 70B-at-64K store that panicked a 128 GB host is refused")
    func longContextStoreOnFullHostIsRefused() {
        let kv = 20 * Self.gib  // 80 layers x 2 x 8 kv-heads x 128 dim x 65536 x fp16
        let active = 92 * Self.gib  // 70 GB weights + the live KV, already resident
        #expect(
            !CacheStoreBudget.canStore(
                cacheBytes: kv, activeBytes: active, budgetBytes: 128 * Self.gib,
                policy: Self.safeAuto),
            "storing this cache needs tens of GiB the host does not have — it must be skipped"
        )
    }

    /// The guard must not disable the prefix cache for ordinary work. A normal chat
    /// turn's cache is a rounding error against the budget.
    @Test("An ordinary cache on a healthy host still stores")
    func ordinaryStoreIsAllowed() {
        #expect(
            CacheStoreBudget.canStore(
                cacheBytes: 512 * (1 << 20),  // 512 MiB — a few thousand tokens
                activeBytes: 40 * Self.gib,
                budgetBytes: 128 * Self.gib,
                policy: Self.safeAuto))
    }

    /// The same cache that is refused on a full host is fine on an empty one:
    /// the verdict is about headroom, not about the cache being "too big".
    @Test("Headroom, not size, decides")
    func sameCacheFitsWhenTheHostIsEmpty() {
        let kv = 20 * Self.gib
        let budget = 128 * Self.gib
        #expect(
            !CacheStoreBudget.canStore(
                cacheBytes: kv, activeBytes: 92 * Self.gib, budgetBytes: budget,
                policy: Self.safeAuto))
        #expect(
            CacheStoreBudget.canStore(
                cacheBytes: kv, activeBytes: 8 * Self.gib, budgetBytes: budget,
                policy: Self.safeAuto))
    }

    /// A host already past its own physical memory has nothing to hand out. This is
    /// the degenerate end of the headroom calculation and must not underflow into
    /// an accidental "yes".
    @Test("A host already over budget stores nothing")
    func overBudgetHostRefuses() {
        #expect(
            !CacheStoreBudget.canStore(
                cacheBytes: 1 << 20, activeBytes: 130 * Self.gib, budgetBytes: 128 * Self.gib,
                policy: Self.safeAuto))
    }

    /// The store copies the cache once — not three times.
    ///
    /// An earlier revision of this guard budgeted for three materialisations (a deep
    /// copy, a host `Data` extract, and the disk-store cache) and none of the three
    /// exist: `copy()` takes buffer-sharing slices, `extractLayerData` returns
    /// references, `makeDiskStoreCache` is a no-op on the raw path, and the
    /// safetensors writer streams to the fd. The reporter's own numbers say the same
    /// thing — a 23 GiB excursion against a 23 GiB cache. Overcounting is not free:
    /// at 3x this refused stores that fit.
    @Test("The store is budgeted at one copy of the cache, matching the reported excursion")
    func materializationFactorMatchesTheEvidence() {
        #expect(CacheStoreBudget.materializationFactor == 1)
    }

    /// A flat 4 GiB reserve is right on a 128 GB Mac and nonsense on an 8 GB one:
    /// with a 4.5 GiB model resident, `active + 0 + 4 GiB` already exceeds 8 GiB,
    /// so a fixed margin would refuse EVERY store — silently disabling the prefix
    /// cache on the machines whose users most notice a slow re-prefill. A share of
    /// headroom scales on its own.
    @Test("A small Mac running a small model still caches")
    func smallHostStillCaches() {
        // 8 GB Mac, ~4.5 GiB 8B 4-bit model resident, 4k-token KV (~0.5 GiB).
        // Headroom 3.5 GiB, of which Safe Auto lends 25% = 0.875 GiB.
        let active = 9 * Self.gib / 2
        #expect(
            CacheStoreBudget.canStore(
                cacheBytes: Self.gib / 2, activeBytes: active, budgetBytes: 8 * Self.gib,
                policy: Self.safeAuto),
            "an 8 GB host must not have its prefix cache silently disabled")

        // 16 GB Mac, same model, a much longer 32k context (~4 GiB of KV): headroom
        // is 11.5 GiB and Safe Auto lends 2.875 GiB, so this one is still refused.
        #expect(
            !CacheStoreBudget.canStore(
                cacheBytes: 4 * Self.gib, activeBytes: active, budgetBytes: 16 * Self.gib,
                policy: Self.safeAuto))
    }

    /// The guard must not quietly disable the prefix cache for the long-context
    /// requests that most need it. With a 94 GiB pack resident on a 128 GB Mac, a
    /// multi-thousand-token conversation still caches — it is only the runaway case
    /// that is refused.
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
                    budgetBytes: physical,
                    policy: Self.safeAuto),
                "a \(tokens)-token cache must still be storable on a healthy host"
            )
        }
    }

    /// The whole point of the change: the user's memory-safety level has to reach
    /// this decision. It used to not — the budget was raw physical RAM and a
    /// hard-coded margin, identical whether the user asked for Performance or
    /// Strict. Same host, same model, same cache; only the safety level moves.
    @Test("The user's safety level actually changes the verdict")
    func safetyLevelIsHonoured() {
        let physical = 128 * Self.gib
        let active = 96 * Self.gib  // headroom = 32 GiB
        let kv = 10 * Self.gib  // 31% of headroom

        // Strict (15%) and Safe Auto (25%) both refuse a cache this size...
        #expect(
            !CacheStoreBudget.canStore(
                cacheBytes: kv, activeBytes: active, budgetBytes: physical, policy: .strict))
        #expect(
            !CacheStoreBudget.canStore(
                cacheBytes: kv, activeBytes: active, budgetBytes: physical, policy: .safeAuto))
        // ...while Performance (45%) and the diagnostic mode (55%) allow it.
        #expect(
            CacheStoreBudget.canStore(
                cacheBytes: kv, activeBytes: active, budgetBytes: physical, policy: .performance))
        #expect(
            CacheStoreBudget.canStore(
                cacheBytes: kv, activeBytes: active, budgetBytes: physical,
                policy: .diagnosticDangerous))
    }

    /// The levels have to be ordered the way their names promise, or the slider is
    /// decorative in a new way. Strictly monotonic: no two levels are the same gate.
    @Test("Stricter levels are strictly stricter")
    func levelsAreMonotonic() {
        let ordered: [CacheStorePolicy] = [
            .strict, .safeAuto, .balanced, .performance, .diagnosticDangerous,
        ]
        for (looser, tighter) in zip(ordered.dropFirst(), ordered) {
            #expect(
                tighter.headroomFraction < looser.headroomFraction,
                "each level must lend strictly more headroom than the one below it")
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

/// The "Safety Level" slider osaurus renders was wired to a field nothing read.
@Suite("Memory-safety level")
struct MemorySafetyLevelTests {

    /// The bug, pinned: setting the slider used to leave `mode` — the thing every
    /// resolver actually switches on — untouched, so the control changed nothing.
    @Test("Moving the safety slider moves the safety mode")
    func sliderDrivesMode() {
        var settings = VMLXMemorySafetySettings()  // Safe Auto
        #expect(settings.slider == 2)

        settings.slider = 3
        #expect(settings.mode == .strict, "the slider must move the mode, not a field beside it")

        settings.slider = 0
        #expect(settings.mode == .performance)

        // And the reverse: the mode picker and the slider can never disagree.
        settings.mode = .diagnosticDangerous
        #expect(settings.slider == 4)
    }

    /// Out-of-range input clamps rather than trapping — it used to be validated as
    /// an error, which is no longer expressible.
    @Test("An out-of-range level clamps")
    func sliderClamps() {
        var settings = VMLXMemorySafetySettings()
        settings.slider = 99
        #expect(settings.mode == .diagnosticDangerous)
        settings.slider = -1
        #expect(settings.mode == .performance)
    }

    /// Existing users must not have their safety level shift under them on upgrade.
    /// Persisted settings that disagree (they dragged the slider back when it did
    /// nothing) keep the mode the engine has actually been enforcing.
    @Test("A persisted slider never overrides the persisted mode")
    func decodingPrefersModeOverStaleSlider() throws {
        // Written by a build where the slider was stored and inert: the user dragged
        // it to Performance (0) but the engine kept running Strict.
        let json =
            #"{"mode":"strict","slider":0,"allowExperimentalMLXPress":false,"failClosedWhenEstimateUnknown":false}"#
        let decoded = try JSONDecoder().decode(
            VMLXMemorySafetySettings.self, from: Data(json.utf8))
        #expect(decoded.mode == .strict, "the level actually in force must survive the upgrade")
        #expect(decoded.slider == 3, "and the number shown for it must now tell the truth")
    }

    /// The persisted/API shape does not change: `slider` is still emitted.
    @Test("The encoded shape still carries the slider")
    func encodingStillEmitsSlider() throws {
        let data = try JSONEncoder().encode(VMLXMemorySafetySettings(mode: .balanced))
        let object = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["slider"] as? Int == 1)
        #expect(object["mode"] as? String == "balanced")
    }

    /// Each level has to reach the cache-store gate, which is the decision the
    /// load-time caps cannot make.
    @Test("Each safety level carries a distinct cache-store policy")
    func modesMapToStorePolicies() {
        #expect(VMLXMemorySafetyMode.performance.cacheStorePolicy == .performance)
        #expect(VMLXMemorySafetyMode.safeAuto.cacheStorePolicy == .safeAuto)
        #expect(VMLXMemorySafetyMode.strict.cacheStorePolicy == .strict)
        #expect(VMLXMemorySafetyMode.diagnosticDangerous.cacheStorePolicy == .diagnosticDangerous)
    }
}
