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

/// A model whose prefill positions depend on the WHOLE prompt (M-RoPE: media placeholders occupy
/// compressed positions), so a suffix prefilled after a cache restore must be positioned from the
/// full request, not from the suffix alone. Both schedulers call this once per prefill, right before
/// the first `prepare`, with the request's full input and the number of prompt tokens the cache
/// already holds (0 for a cold prefill). A model must leave cold prefills exactly as they were.
public protocol RestoredPrefixPositionModel: AnyObject {
    func prepareRestoredPrefixPositions(fullInput: LMInput, restoredTokenCount: Int)
}

/// A prefix of the lookup token list whose rows may be keyed under a salt
/// without the media component. `tokenCount` is the index of the first media
/// placeholder token in the probed token space, so a boundary `b` qualifies
/// iff `b <= tokenCount`.
public struct MediaIndependentPrefix: Sendable, Equatable {
    public let salt: String?
    public let tokenCount: Int
    /// Per-media-item salt segments (``CacheBoundarySaltPlan/segments``); empty when only the
    /// pre-media prefix has its own salt.
    public let segments: [CacheBoundarySaltPlan.Segment]

    public init(salt: String?, tokenCount: Int, segments: [CacheBoundarySaltPlan.Segment] = []) {
        self.salt = salt
        self.tokenCount = tokenCount
        self.segments = segments
    }

    /// The salt a boundary is keyed under when it is not the request salt: the pre-media salt at
    /// or before the first media token, or the salt of exactly the media before it when the model
    /// keys boundaries per media item. Nil: the boundary uses the request salt.
    public func salt(forBoundary boundary: Int) -> String?? {
        if let segment = segments.first(where: { $0.contains(boundary) }) { return .some(segment.salt) }
        if boundary <= tokenCount { return .some(salt) }
        return nil
    }
}

/// One media item of a prompt (an image, or a whole video including any text between its frame
/// runs): the token range from its first to last placeholder, and where its pixels sit in that
/// kind's concatenated pixel tensor and frame list.
public struct MediaItemSpan: Sendable, Equatable {
    public enum Kind: Sendable, Equatable { case image, video }
    public let kind: Kind
    public let tokenRange: Range<Int>
    public let pixelRows: Range<Int>
    public let frameIndex: Int

    public init(kind: Kind, tokenRange: Range<Int>, pixelRows: Range<Int>, frameIndex: Int) {
        self.kind = kind
        self.tokenRange = tokenRange
        self.pixelRows = pixelRows
        self.frameIndex = frameIndex
    }
}

/// A model whose state at a boundary depends only on the tokens before it and the media items whose
/// placeholders all lie before it (causal forward, media spliced only at placeholders, positions of
/// earlier tokens independent of later media). Such a boundary is keyed under the salt of exactly
/// that earlier media, so the rows a previous turn stored (with the same earlier media) are restored
/// when a turn adds another image or video, and only the new media is encoded.
public protocol MediaItemPrefixCacheModel: PrefixCacheInvariantModel {
    /// Every media item of `input` in prompt order (indices into `tokenIds`), or nil when the items
    /// cannot be matched to the placeholders exactly; nil keeps the pre-media-only behaviour.
    func mediaItemSpans(in input: LMInput, tokenIds: [Int]) -> [MediaItemSpan]?
}

/// `input` restricted to the media `items` (contiguous in prompt order) with `text`; image and video
/// pixel rows and frames are sliced to exactly those items, so its media salt equals the salt of a
/// request that carried only them.
func mediaSubsetInput(
    _ input: LMInput, items: ArraySlice<MediaItemSpan>, text: LMInput.Text
) -> LMInput {
    func subset(_ kind: MediaItemSpan.Kind, pixels: MLXArray?, frames: [THW]?) -> (MLXArray, [THW])? {
        let chosen = items.filter { $0.kind == kind }
        guard let pixels, let frames, let first = chosen.first, let last = chosen.last else { return nil }
        return (
            pixels[first.pixelRows.lowerBound ..< last.pixelRows.upperBound],
            Array(frames[first.frameIndex ... last.frameIndex]))
    }
    let image = subset(.image, pixels: input.image?.pixels, frames: input.image?.frames)
        .map { LMInput.ProcessedImage(pixels: $0.0, frames: $0.1) }
    let video = subset(.video, pixels: input.video?.pixels, frames: input.video?.frames)
        .map { LMInput.ProcessedVideo(pixels: $0.0, frames: $0.1) }
    return LMInput(
        text: text, image: image, video: video, audio: input.audio,
        mediaTokenIds: input.mediaTokenIds, cacheScopeSalt: input.cacheScopeSalt,
        cachePrefixTokenCounts: input.cachePrefixTokenCounts,
        cacheStablePrefixTokenCounts: input.cacheStablePrefixTokenCounts,
        cachePromptIntent: input.cachePromptIntent, cacheRestorePolicy: input.cacheRestorePolicy,
        toolSchemas: input.toolSchemas,
        canonicalRequiredToolContext: input.canonicalRequiredToolContext)
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
    /// Boundaries `lower ... upper` (between the end of one media item and the start of the next)
    /// keyed under `salt`: the salt of a request carrying exactly the media before them.
    public struct Segment: Sendable, Equatable {
        public let lower: Int
        public let upper: Int
        public let salt: String?
        public func contains(_ boundary: Int) -> Bool { boundary >= lower && boundary <= upper }
    }

    public let requestSalt: String?
    public let mediaIndependentSalt: String?
    /// Index of the first media placeholder token in the probed token list;
    /// 0 when no boundary is media-independent.
    public let mediaIndependentTokenCount: Int
    /// Per-item segments after the first media item (``MediaItemPrefixCacheModel``); empty otherwise.
    public let segments: [Segment]
    /// The request's media items, when the model keys boundaries per item.
    public let itemSpans: [MediaItemSpan]

    public init(requestSalt: String?) {
        self.requestSalt = requestSalt
        self.mediaIndependentSalt = nil
        self.mediaIndependentTokenCount = 0
        self.segments = []
        self.itemSpans = []
    }

    public init(
        requestSalt: String?, mediaIndependentSalt: String?, mediaIndependentTokenCount: Int,
        segments: [Segment] = [], itemSpans: [MediaItemSpan] = []
    ) {
        self.requestSalt = requestSalt
        self.mediaIndependentSalt = mediaIndependentSalt
        self.mediaIndependentTokenCount = max(0, mediaIndependentTokenCount)
        self.segments = segments
        self.itemSpans = itemSpans
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
        if let itemModel = model as? any MediaItemPrefixCacheModel,
            let spans = itemModel.mediaItemSpans(in: input, tokenIds: lookupTokenIds),
            let firstSpan = spans.first, firstSpan.tokenRange.lowerBound == first
        {
            var segments: [Segment] = []
            for k in 1 ..< spans.count {
                let earlier = mediaSubsetInput(input, items: spans[0 ..< k], text: input.text)
                segments.append(Segment(
                    lower: spans[k - 1].tokenRange.upperBound,
                    upper: spans[k].tokenRange.lowerBound,
                    salt: computeCacheSalt(for: earlier, parameters: parameters)))
            }
            return CacheBoundarySaltPlan(
                requestSalt: requestSalt,
                mediaIndependentSalt: computeCacheSalt(for: input, parameters: parameters, includingMedia: false),
                mediaIndependentTokenCount: first,
                segments: segments,
                itemSpans: spans)
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
        if let segment = segments.first(where: { $0.contains(tokenCount) }) {
            return segment.salt
        }
        return requestSalt
    }

    /// What the coordinator's fetch needs to also probe the media-independent
    /// rows; nil when every boundary uses `requestSalt`.
    public var mediaIndependentPrefix: MediaIndependentPrefix? {
        guard mediaIndependentTokenCount > 0 else { return nil }
        return MediaIndependentPrefix(
            salt: mediaIndependentSalt, tokenCount: mediaIndependentTokenCount, segments: segments)
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
        guard mediaIndependentTokenCount > 0, matchedTokens > 0 else { return false }
        // Per-item keying also resumes between two items: every item of the suffix is complete
        // there, and `mediaCarryingSuffixInput` hands `prepare` exactly those items.
        return matchedTokens <= mediaIndependentTokenCount
            || segments.contains { $0.contains(matchedTokens) }
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
    from input: LMInput, remainingTokens: [Int], matchedTokens: Int,
    itemSpans: [MediaItemSpan] = []
) -> LMInput {
    let tokens = MLXArray(remainingTokens.map(Int32.init)).expandedDimensions(axis: 0)
    var suffixMask: MLXArray?
    if let mask = input.text.mask, mask.size == matchedTokens + remainingTokens.count {
        let flat = mask.reshaped(-1)[matchedTokens...]
        suffixMask = mask.ndim >= 2 ? flat[.newAxis, 0...] : flat
    }
    let text = LMInput.Text(tokens: tokens, mask: suffixMask, tokenIds: remainingTokens)
    if let firstInSuffix = itemSpans.firstIndex(where: { $0.tokenRange.lowerBound >= matchedTokens }),
        firstInSuffix > 0
    {
        // Items before the boundary are already in the restored state: encode only the suffix's.
        return mediaSubsetInput(input, items: itemSpans[firstInSuffix...], text: text)
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
