// Copyright © 2025 Apple Inc. All rights reserved.

import Foundation

/// Configuration for ``CacheCoordinator``, controlling which cache tiers
/// are enabled and their sizing parameters.
///
/// ## KV-sizing contract
///
/// As of 2026-04-21 the coordinator owns KV cache sizing end-to-end. Two
/// fields drive the policy; callers can set either, both, or neither:
///
/// - ``defaultKVMode``: applied to any admitted slot whose
///   ``GenerateParameters.kvMode`` is `.none`. Use this to make
///   ``KVQuantizationMode/turboQuant(keyBits:valueBits:)`` the default
///   KV representation for every request without touching per-request
///   call sites.
///
/// - ``defaultMaxKVSize``: applied to any admitted slot whose
///   ``GenerateParameters.maxKVSize`` is `nil` **and** whose prompt
///   exceeds ``defaultMaxKVSize`` by more than ``longPromptMultiplier`` ×.
///   When that happens the slot is configured to use
///   ``RotatingKVCache`` with a fixed window instead of unbounded
///   ``KVCacheSimple``. This keeps worst-case KV memory bounded even
///   when a long document prompt arrives through an API boundary that
///   did not pass `maxKVSize` explicitly.
///
/// Callers always win: explicit per-request `kvMode` and `maxKVSize` are
/// honored untouched. The coordinator defaults only fill in the gaps.
public struct CacheCoordinatorConfig: Sendable {

    /// Whether the in-memory paged KV cache is enabled.
    public var usePagedCache: Bool

    /// Whether the on-disk L2 cache (SQLite + safetensors) is enabled.
    public var enableDiskCache: Bool

    /// Number of tokens per paged cache block.
    public var pagedBlockSize: Int

    /// Maximum number of blocks in the paged cache pool (including sentinel).
    public var maxCacheBlocks: Int

    /// Maximum disk cache size in gigabytes.
    public var diskCacheMaxGB: Float

    /// Directory for disk cache files. If `nil`, a default temp directory is used.
    public var diskCacheDir: URL?

    /// Maximum number of SSM state entries in the companion LRU cache.
    public var ssmMaxEntries: Int

    /// Whether hybrid SSM models should run the prompt-boundary companion
    /// state re-derive/store pass after generation.
    ///
    /// Keep this enabled for production hybrid-cache reuse. Disable only for
    /// controlled A/B matrix rows where the host intentionally wants to prove
    /// the fallback full-prefill path or isolate non-SSM cache behavior.
    public var enableSSMReDerive: Bool

    /// Model-specific key to prevent cross-model cache poisoning.
    /// Include model path, type, or a unique identifier. When set, cache hashes
    /// incorporate this key so different models with the same tokenizer cannot
    /// return each other's cached KV state.
    public var modelKey: String?

    /// Default ``KVQuantizationMode`` applied to admitted slots whose
    /// request ``GenerateParameters.kvMode`` is `.none`.
    ///
    /// When non-`.none`, every request that did not explicitly pick a
    /// KV quantization mode will run with this mode. The most useful
    /// value is ``KVQuantizationMode/turboQuant(keyBits:valueBits:)``
    /// which provides ~5× KV memory savings with negligible quality
    /// degradation — suitable as a global default for host applications
    /// that want memory-bounded inference across arbitrary user prompts.
    ///
    /// Leave as `.none` to preserve current behavior (float KV by default).
    public var defaultKVMode: KVQuantizationMode

    /// Default ``GenerateParameters.maxKVSize`` applied to slots that
    /// (a) did not set `maxKVSize`, and (b) have prompts exceeding
    /// ``longPromptMultiplier`` × this value.
    ///
    /// When applied, the slot's cache is allocated as ``RotatingKVCache``
    /// with a window equal to this value, bounding KV memory at a fixed
    /// amount regardless of prompt length. Sliding-window models that
    /// already allocate `RotatingKVCache` are unaffected.
    ///
    /// Set to `nil` to disable the default. The default-on threshold is
    /// intentionally gated on prompt length: short chat turns never
    /// take the rotating-window hit from an unintended global cap.
    public var defaultMaxKVSize: Int?

    /// Multiplier applied to ``defaultMaxKVSize`` to decide whether a
    /// prompt is "long" and should inherit the default KV size cap.
    ///
    /// A value of `2.0` means: only when `promptTokens.count > 2 ×
    /// defaultMaxKVSize` does the slot adopt the cap. Lower values make
    /// the default kick in sooner; higher values preserve full-attention
    /// behavior for more workloads. Ignored when `defaultMaxKVSize` is nil.
    public var longPromptMultiplier: Double

    public init(
        usePagedCache: Bool = false,
        enableDiskCache: Bool = false,
        pagedBlockSize: Int = 64,
        maxCacheBlocks: Int = 1000,
        diskCacheMaxGB: Float = 10.0,
        diskCacheDir: URL? = nil,
        ssmMaxEntries: Int = 50,
        enableSSMReDerive: Bool = true,
        modelKey: String? = nil,
        defaultKVMode: KVQuantizationMode = .none,
        defaultMaxKVSize: Int? = nil,
        longPromptMultiplier: Double = 2.0
    ) {
        self.usePagedCache = usePagedCache
        self.enableDiskCache = enableDiskCache
        self.pagedBlockSize = pagedBlockSize
        self.maxCacheBlocks = maxCacheBlocks
        self.diskCacheMaxGB = diskCacheMaxGB
        self.diskCacheDir = diskCacheDir
        self.ssmMaxEntries = ssmMaxEntries
        self.enableSSMReDerive = enableSSMReDerive
        self.modelKey = modelKey
        self.defaultKVMode = defaultKVMode
        self.defaultMaxKVSize = defaultMaxKVSize
        self.longPromptMultiplier = longPromptMultiplier
    }
}

// MARK: - Policy resolution

extension CacheCoordinatorConfig {

    /// Resolve the effective ``KVQuantizationMode`` and ``maxKVSize`` for a
    /// request, applying the coordinator's defaults where the request did
    /// not set them explicitly.
    ///
    /// Explicit request values ALWAYS win. This function only fills gaps.
    ///
    /// - Parameters:
    ///   - kvMode: the request's ``GenerateParameters.kvMode``.
    ///   - maxKVSize: the request's ``GenerateParameters.maxKVSize``.
    ///   - promptTokenCount: number of tokens in the request's prompt.
    /// - Returns: the effective `(kvMode, maxKVSize)` after applying defaults.
    public func resolveKVPolicy(
        kvMode: KVQuantizationMode,
        maxKVSize: Int?,
        promptTokenCount: Int
    ) -> (kvMode: KVQuantizationMode, maxKVSize: Int?) {
        var effectiveMode = kvMode
        if case .none = kvMode, case .none = defaultKVMode {
            // Both request and default are .none — nothing to fill in.
        } else if case .none = kvMode {
            effectiveMode = defaultKVMode
        }

        var effectiveMax = maxKVSize
        if maxKVSize == nil,
           let cap = defaultMaxKVSize,
           Double(promptTokenCount) > Double(cap) * longPromptMultiplier
        {
            effectiveMax = cap
        }

        return (effectiveMode, effectiveMax)
    }
}
