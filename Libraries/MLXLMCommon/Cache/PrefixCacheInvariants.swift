// Copyright © 2026 Osaurus AI
//
// Prefix-cache invariants a model declares about its own forward pass, and
// the per-boundary cache salt they unlock.
//
// Nothing here is inferred from a family name, a cache topology or a config
// value: a model that does not adopt the protocol gets `false` for every
// invariant and keeps the conservative behaviour exactly.

import Foundation
import MLX

/// Properties of a model's forward pass that let the prefix cache reuse rows
/// it would otherwise have to leave alone. Each property names the invariant
/// the model vouches for; adopt the protocol only with evidence for each.
public protocol PrefixCacheInvariantModel: AnyObject {
    /// The KV/recurrent state of every position strictly before the first
    /// media placeholder token equals the text-only prompt's state at that
    /// position: the forward is causal, nothing before that token depends on
    /// the media (no media-derived positions or rotations, no prepended media
    /// state), and media embeddings are spliced only at placeholder positions.
    ///
    /// When true, a request that carries media looks up and stores every
    /// prompt boundary that ends at or before the first media token under the
    /// text-only salt, so the rows an earlier text turn wrote are reused and a
    /// later text turn can reuse the rows a media turn writes. Boundaries at
    /// or after the first media token keep the media salt.
    ///
    /// Must be false for M-RoPE families (Qwen-VL class) and for any model
    /// whose pre-media positions or states depend on the media.
    var statesBeforeFirstMediaTokenAreMediaIndependent: Bool { get }

    /// The cache state restored from an exact token-prefix row is the state a
    /// cold prefill of those tokens produces, whatever the request appends
    /// after the prefix: recurrent state included, and independent of the
    /// suffix the request will prefill next (a forced tool-selection tail is
    /// just another suffix).
    ///
    /// When true, `CacheRestorePolicy.freshRequiredToolSelection` keeps the
    /// ordinary longest-prefix restore for this model's disk-backed topology
    /// instead of skipping the cache fetch and prefilling the whole prompt.
    var exactPrefixRestoreIsSuffixIndependent: Bool { get }
}

/// A prefix of the lookup token list whose rows may be keyed under a salt
/// without the media component. `tokenCount` is the index of the first media
/// placeholder token in the probed token space, so a boundary `b` qualifies
/// iff `b <= tokenCount`.
public struct MediaIndependentPrefix: Sendable, Equatable {
    public let salt: String?
    public let tokenCount: Int

    public init(salt: String?, tokenCount: Int) {
        self.salt = salt
        self.tokenCount = tokenCount
    }
}

/// The cache salt a request uses per prompt boundary.
///
/// `requestSalt` is what `computeCacheSalt(for:parameters:)` returns for the
/// request (media + scope + KV policy). For a media request on a model that
/// declares ``PrefixCacheInvariantModel/statesBeforeFirstMediaTokenAreMediaIndependent``,
/// boundaries at or before the first media token use `mediaIndependentSalt`
/// instead: the salt the same request would have without its media (same
/// scope, same KV policy), which is exactly what an earlier text-only turn of
/// the chat used. Text-only requests and every other model get a plan whose
/// every boundary uses `requestSalt`.
public struct CacheBoundarySaltPlan: Sendable, Equatable {
    public let requestSalt: String?
    public let mediaIndependentSalt: String?
    /// Index of the first media placeholder token in the probed token list;
    /// 0 when no boundary is media-independent.
    public let mediaIndependentTokenCount: Int

    public init(requestSalt: String?) {
        self.requestSalt = requestSalt
        self.mediaIndependentSalt = nil
        self.mediaIndependentTokenCount = 0
    }

    public init(requestSalt: String?, mediaIndependentSalt: String?, mediaIndependentTokenCount: Int) {
        self.requestSalt = requestSalt
        self.mediaIndependentSalt = mediaIndependentSalt
        self.mediaIndependentTokenCount = max(0, mediaIndependentTokenCount)
    }

    /// Build the plan for a request. `lookupTokenIds` is the token list the
    /// coordinator probes (the raw prompt, or the post-prepare alias when the
    /// model prunes placeholders), so the first-media index is computed in the
    /// space the probe uses. The pre-media prefix is identical in both spaces.
    public static func make(
        requestSalt: String?,
        input: LMInput,
        lookupTokenIds: [Int],
        model: any LanguageModel,
        parameters: GenerateParameters
    ) -> CacheBoundarySaltPlan {
        guard input.hasMediaContent,
              (model as? PrefixCacheInvariantModel)?
                  .statesBeforeFirstMediaTokenAreMediaIndependent == true,
              let mediaTokenIds = input.mediaTokenIds, !mediaTokenIds.isEmpty
        else { return CacheBoundarySaltPlan(requestSalt: requestSalt) }
        let media = Set(mediaTokenIds)
        guard let first = lookupTokenIds.firstIndex(where: media.contains), first > 0 else {
            return CacheBoundarySaltPlan(requestSalt: requestSalt)
        }
        return CacheBoundarySaltPlan(
            requestSalt: requestSalt,
            mediaIndependentSalt: computeCacheSalt(for: input, parameters: parameters, includingMedia: false),
            mediaIndependentTokenCount: first)
    }

    /// The salt a boundary of `tokenCount` prompt tokens is keyed under.
    public func salt(forBoundary tokenCount: Int) -> String? {
        if mediaIndependentTokenCount > 0, tokenCount <= mediaIndependentTokenCount {
            return mediaIndependentSalt
        }
        return requestSalt
    }

    /// What the coordinator's fetch needs to also probe the media-independent
    /// rows; nil when every boundary uses `requestSalt`.
    public var mediaIndependentPrefix: MediaIndependentPrefix? {
        guard mediaIndependentTokenCount > 0 else { return nil }
        return MediaIndependentPrefix(salt: mediaIndependentSalt, tokenCount: mediaIndependentTokenCount)
    }
}

extension CacheBoundarySaltPlan {
    /// Whether a partial cache hit at `matchedTokens` may resume even though
    /// media placeholder tokens remain in the suffix. True only for a
    /// media-independent prefix (the model declared
    /// ``PrefixCacheInvariantModel/statesBeforeFirstMediaTokenAreMediaIndependent``)
    /// and a boundary at or before the first media token: the suffix then
    /// holds every placeholder run complete, so `prepare` splices the media
    /// at exactly the positions a cold prefill would. A boundary inside or
    /// after a run keeps the conservative rollback.
    public func resumeBeforeMediaIsSafe(matchedTokens: Int) -> Bool {
        mediaIndependentTokenCount > 0 && matchedTokens > 0
            && matchedTokens <= mediaIndependentTokenCount
    }
}

/// Whether a partial hit must roll back to a full prefill because media
/// placeholders remain in its suffix. The pre-media resume above is the one
/// exemption; opaque post-prepare token spaces (video pruning) keep the
/// rollback because the suffix tokens are not what `prepare` consumes.
public func cacheHitSuffixRequiresMediaRollback(
    input: LMInput, remainingTokens: [Int], matchedTokens: Int, plan: CacheBoundarySaltPlan
) -> Bool {
    guard input.cacheHitSuffixContainsMediaPlaceholder(remainingTokens) else { return false }
    guard !input.requiresPostPrepareCacheKey else { return true }
    return !plan.resumeBeforeMediaIsSafe(matchedTokens: matchedTokens)
}

/// The suffix a pre-media partial hit prefills: the remaining tokens with the
/// request's media tensors and placeholder ids attached (the complete
/// placeholder runs are in this suffix, so the splice needs them), the mask
/// sliced to the suffix, and the request's cache metadata carried over.
public func mediaCarryingSuffixInput(
    from input: LMInput, remainingTokens: [Int], matchedTokens: Int
) -> LMInput {
    let tokens = MLXArray(remainingTokens.map(Int32.init)).expandedDimensions(axis: 0)
    var suffixMask: MLXArray?
    if let mask = input.text.mask, mask.size == matchedTokens + remainingTokens.count {
        let flat = mask.reshaped(-1)[matchedTokens...]
        suffixMask = mask.ndim >= 2 ? flat[.newAxis, 0...] : flat
    }
    return LMInput(
        text: .init(tokens: tokens, mask: suffixMask, tokenIds: remainingTokens),
        image: input.image, video: input.video, audio: input.audio,
        mediaTokenIds: input.mediaTokenIds, cacheScopeSalt: input.cacheScopeSalt,
        cachePrefixTokenCounts: input.cachePrefixTokenCounts,
        cacheStablePrefixTokenCounts: input.cacheStablePrefixTokenCounts,
        cachePromptIntent: input.cachePromptIntent, cacheRestorePolicy: input.cacheRestorePolicy,
        toolSchemas: input.toolSchemas,
        canonicalRequiredToolContext: input.canonicalRequiredToolContext)
}

/// Whether `CacheRestorePolicy.freshRequiredToolSelection` must skip the
/// disk-backed restore for this model and cache, or may keep the ordinary
/// longest-prefix fetch. The skip was introduced for topologies whose
/// path-dependent restore had no proof; a model that declares
/// ``PrefixCacheInvariantModel/exactPrefixRestoreIsSuffixIndependent`` carries
/// that proof and keeps reuse. Exact-prefix caches never skipped.
public func requiredToolSelectionSkipsDiskBackedRestore(
    model: any LanguageModel,
    cache: [any KVCache],
    restorePolicy: LMInput.CacheRestorePolicy
) -> Bool {
    guard restorePolicy == .freshRequiredToolSelection,
          cacheRequiresDiskBackedCoordinatorRestore(cache),
          !cacheRestoreIsExactPrefix(cache)
    else { return false }
    return (model as? PrefixCacheInvariantModel)?.exactPrefixRestoreIsSuffixIndependent != true
}
