// Copyright © 2026 Jinho Jang. All rights reserved.
//
// JangPressLoadOptions — legacy spelling for MLXPress (axis E,
// cold-weight tier). Default-off; consumers (osaurus, JANG Studio,
// CLI tools) explicitly construct an instance and pass it through
// `loadWeights(...)` to enable the feature.
//
// Prefer the MLXPressLoadOptions typealias and MLXPRESS_* env vars for new
// callers. The JangPress spelling remains source-compatible for existing
// Osaurus/vmlx integrations.

import Foundation

public struct JangPressLoadOptions: Sendable, Equatable {
    /// Master switch. Default `false` — MLXPress is opt-in.
    public var enabled: Bool

    /// 0..100 — % of routed-MoE weight mass open to compaction during
    /// quiesce. `0` arms the failsafe controller without compacting
    /// anything (kernel reclaim under pressure still works);
    /// `70` is the production-recommended value for tight hosts;
    /// `100` keeps only the top-k hot expert set pinned.
    public var compressPct: Int

    /// Backend selection. `.mmap` is the production default
    /// (file-backed, page-cache shared with MLX, zero RAM doubling on
    /// patched osaurus mlx-swift pins). `.none` disables the routed-
    /// expert tier even if `enabled == true` (still arms the
    /// embed/lm_head Zipfian tier).
    public var backend: Backend

    /// Eviction aggressiveness for the `.mmap` backend.
    /// `.soft` issues `madvise(MADV_DONTNEED)` — kernel HINTS, ignored
    /// when free RAM is plentiful. **Failsafe default.**
    /// `.force` issues `msync(MS_INVALIDATE)` — kernel drops pages
    /// immediately. Use only on memory-constrained hosts where eager
    /// reclaim is required and cold-fault latency is acceptable.
    public var forceMode: ForceMode

    /// Pre-fault top-`hotPercent` of tiles at arm time. Defaults to
    /// `true`. Disabling adds within-process drift at temperature 0
    /// (see JANGPRESS-DEEP-TRACE Issue 5 for the full analysis).
    public var enablePrefetch: Bool

    /// Enable router-aware canonical mmap advice. When true, routed-MoE
    /// layers can report the exact expert ids selected during decode;
    /// MLXPress then issues `MADV_WILLNEED` for newly hot experts and
    /// `MADV_DONTNEED` for older experts beyond the per-layer hot-set
    /// budget. This is precise OS page advice for mmap-backed canonical
    /// weights, not a custom compressed blob codec.
    ///
    /// Default is `false` because the first CPU-readback implementation
    /// is correct but too slow for production decode. Set explicitly or
    /// export `MLXPRESS_ROUTER_ADVICE=1` for experiments.
    public var enableRouterAdvice: Bool

    public enum Backend: String, Sendable, Equatable {
        /// File-backed mmap probe/status plus canonical MLX mmap advice
        /// when the patched osaurus mlx-swift safetensors loader ABI is
        /// present.
        case mmap
        /// Disable the routed-expert tier entirely (the embed tier may
        /// still be co-instantiated).
        case none
    }

    public enum ForceMode: String, Sendable, Equatable {
        case soft, force
    }

    /// Sane production-grade default — feature off, ready to be turned
    /// on per-call via the constructor parameters below.
    // SPEED + PRODUCT CONTRACT (Eric): JangPress / MLXPress stay OFF by default and are never
    // enabled in benches or speed proofs. Compressed / pressure-evicted expert tiers make MTP verify
    // (2-8 rows touching many experts) page in from storage, which is exactly the in-app MTP collapse
    // pattern. Do not change this default or route Flash-Next / Allosaurus / 27B through it.
    public static let disabled = JangPressLoadOptions()

    public init(
        enabled: Bool = false,
        compressPct: Int = 70,
        backend: Backend = .mmap,
        forceMode: ForceMode = .soft,
        enablePrefetch: Bool = true,
        enableRouterAdvice: Bool = false
    ) {
        self.enabled = enabled
        self.compressPct = max(0, min(100, compressPct))
        self.backend = backend
        self.forceMode = forceMode
        self.enablePrefetch = enablePrefetch
        self.enableRouterAdvice = enableRouterAdvice
    }
}

/// MLXPress runtime handles attached to a `ModelContext` after
/// `loadWeights(...)`.
///
/// **State of play (2026-05-03)**: the original controller-driven
/// `acquire/release` tier was deleted because it only advised a
/// parallel probe mapping while MLX's canonical heap copies stayed
/// resident. The production savings path is now the patched
/// osaurus mlx-swift safetensors loader: canonical MLX arrays can be
/// backed by mmap file pages, and vmlx can advise those exact routed
/// expert ranges warm/cold through a C ABI.
///
/// What remains:
///   * `mmap` — kept as a tile-classification probe/status utility.
///     Canonical storage and reclaim happen in the patched MLX loader,
///     not in this probe object.
///   * `embed` — Zipfian embed/lm_head cache, orthogonal to the
///     routed-expert tier and still useful on its own.
///   * `JangPressCanonicalExpertAdvisor` — legacy internal name for the
///     MLXPress global decode-time policy
///     that reports router-selected experts to the patched MLX mmap
///     registry for `MADV_WILLNEED` / `MADV_DONTNEED`.
public struct JangPressRuntime: Sendable {
    public var mmap: JangPressMmapTier?
    public var embed: JangPressEmbedTier?

    /// The options that produced this runtime. `nil` for `.none`.
    /// Surfaced via ``MLXPressStatus/coldFraction`` so settings UIs
    /// can display the resolved configuration instead of just "—".
    public var appliedOptions: JangPressLoadOptions?

    public init(
        mmap: JangPressMmapTier? = nil,
        embed: JangPressEmbedTier? = nil,
        appliedOptions: JangPressLoadOptions? = nil
    ) {
        self.mmap = mmap
        self.embed = embed
        self.appliedOptions = appliedOptions
    }

    public static let none = JangPressRuntime()

    /// True iff at least one tier is attached. With the controller
    /// gone, this is purely informational — no state machine is
    /// driven by it.
    public var isActive: Bool {
        mmap != nil || embed != nil
    }

    /// Feed prompt token ids into the embed/lm_head Zipfian tier and,
    /// once enough distinct ids have been observed, schedule cold-row
    /// advice off the request hot path. Safe to call unconditionally;
    /// it is a no-op when MLXPress or the embed tier is disabled.
    public func recordPromptTokenActivity(_ tokenIds: [Int]) {
        guard let embed, !tokenIds.isEmpty else { return }
        embed.recordTokenActivity(tokenIds)

        let stats = embed.snapshotIfBuilt()
        guard stats.distinctTokensSeen >= 256 else { return }
        Task.detached(priority: .background) { [weak embed] in
            embed?.applyZipfianAdvise()
        }
    }
}
