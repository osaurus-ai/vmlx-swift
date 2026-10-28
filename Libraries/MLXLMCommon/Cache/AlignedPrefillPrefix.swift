// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX

// Aligned-chunk prefill: where a prompt prefill is cut, and which restored prefix rows a request may reuse, so that a
// prefix-cache hit continues to exactly the state and logits a cold prefill of the same prompt produces.
//
// Why. MLX evaluates a prompt chunk as one matrix, and the same token row comes out a few ulps different inside chunks
// of different lengths. A model whose forward turns those ulps into discrete choices (MiniMax-M3's MSA indexer picks
// sparse blocks from them) then answers differently after a cache hit than without one. A cached row is only
// interchangeable with a cold prefill's row when both were computed inside a chunk of the same ABSOLUTE geometry.
//
// The rule (the port of the vMLX Python runtime's `models/minimax_m3/prefill_geometry.py`), one for every cache tier,
// cold prefill and continuation alike:
//
// * every prompt prefill is cut at each absolute multiple of the prefill step `C` and at the request's store boundary
//   `K` (the generation-prompt-stripped prompt minus its last token, see `storeBoundary`). A continuation after a
//   restore at `r` therefore runs exactly the chunks a cold prefill of the same prompt runs from `r` on;
// * rows inside a full aligned chunk `[j*C, (j+1)*C)` depend only on the tokens up to the chunk end, so any longer
//   prompt with the same prefix may reuse them: reuse stops at a chunk boundary;
// * rows of the partial chunk that ends at `K` were computed in a chunk only a request with the SAME `K` runs, so they
//   are reusable only by an exact-geometry request (reuse == its own `K`: exact repeats).
//
// Rows of one same-shape chunk were measured (Python) to be independent of the other rows' content, but that does not
// let a hit stop mid-chunk: the continuation would then run the rest of that chunk as a shorter matrix.
//
// The Swift cache tiers that hold such a model's state are whole-prompt entries (the L2 disk tier; the paged tier
// cannot carry these caches). Each stored state carries the geometry it was produced in (`AlignedPrefillTag`: `C` and
// the producing request's `K`) inside its own serialized metadata, so ANY stored boundary (the prompt, prompt - 1, a
// message boundary, the post-answer row) can be cut back to what it really holds: rows below the last chunk boundary
// at or before the producing `K`, or all `K` rows for a request with the same `K`. The cache salt carries the schema
// and `C`, so entries from another step, or written before aligned prefill, are never served.
//
// `VMLX_M3_ALIGNED_PREFIX=0` restores the previous behaviour (prefill chunked relative to the restored offset, any
// stored prefix reused whole) for comparisons.

/// The aligned-prefill rule: pure functions of positions, no model state.
public enum AlignedPrefillGeometry {
    /// Identity of the rule. Part of the cache salt and of every stored state's tag.
    public static let schema = "aligned-prefill-v1"
    /// `VMLX_M3_ALIGNED_PREFIX=0` (or false / off / no) turns aligned prefill off.
    public static let environmentKey = "VMLX_M3_ALIGNED_PREFIX"

    public static func isEnabled(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        let value = (environment[environmentKey] ?? "1").trimmingCharacters(in: .whitespaces).lowercased()
        return !["0", "false", "off", "no"].contains(value)
    }

    /// Absolute `[a, b)` pieces that prefill `[start, end)`: cut at every multiple of `step` (not for an atomic media
    /// prefill) and at `boundary`, when they fall strictly inside.
    public static func pieces(
        start: Int, end: Int, step: Int, boundary: Int?, atomic: Bool = false
    ) -> [Range<Int>] {
        guard end > start else { return [] }
        var cuts: Set<Int> = [start, end]
        if let boundary, start < boundary, boundary < end { cuts.insert(boundary) }
        if !atomic {
            let step = max(1, step)
            var cut = (start / step + 1) * step
            while cut < end {
                cuts.insert(cut)
                cut += step
            }
        }
        let ordered = cuts.sorted()
        return zip(ordered.dropLast(), ordered.dropFirst()).map { $0 ..< $1 }
    }

    /// Start of the chunk that ends at `boundary` (== `boundary` when it is itself a chunk boundary).
    public static func trailingChunkStart(boundary: Int, step: Int) -> Int {
        let step = max(1, step)
        return (max(0, boundary) / step) * step
    }

    /// The longest restored prefix a request whose own store boundary is `boundary` may use out of `matched` restored
    /// rows: its boundary when the rows were produced with that same boundary (`exactGeometry`), else the largest chunk
    /// boundary <= min(matched, boundary) (0 for an atomic media request). `granule` rounds the chunk boundary down
    /// further (block-granular tiers that cannot cut inside a block; lcm with the step).
    public static func reuseLimit(
        matched: Int, boundary: Int, step: Int, exactGeometry: Bool, atomic: Bool = false, granule: Int = 1
    ) -> Int {
        guard matched > 0, boundary > 0 else { return 0 }
        if exactGeometry && matched == boundary { return boundary }
        if atomic { return 0 }
        var unit = max(1, step)
        if granule > 1 { unit = unit / gcd(unit, granule) * granule }
        return (min(matched, boundary) / unit) * unit
    }

    /// The store boundary `K` of a request: the prompt with its generation-prompt suffix stripped (the template's
    /// suffix when the prompt ends with it, else the processor's latest history boundary, else nothing stripped),
    /// minus its final token. A cold prefill is cut here, so an exact repeat restores all `K` rows and re-runs only
    /// the last `N - K` tokens. nil when no row would be left (a one-token prompt).
    public static func storeBoundary(
        promptTokenIds: [Int], cachePrefixTokenCounts: [Int], generationPromptSuffix: [Int]
    ) -> Int? {
        let n = promptTokenIds.count
        var stripped = n
        if !generationPromptSuffix.isEmpty, generationPromptSuffix.count < n,
            promptTokenIds.suffix(generationPromptSuffix.count).elementsEqual(generationPromptSuffix)
        {
            stripped = n - generationPromptSuffix.count
        } else if let history = cachePrefixTokenCounts.filter({ $0 > 0 && $0 < n }).max() {
            stripped = history
        }
        let boundary = stripped - 1
        return boundary > 0 ? boundary : nil
    }

    private static func gcd(_ a: Int, _ b: Int) -> Int {
        var (a, b) = (abs(a), abs(b))
        while b != 0 { (a, b) = (b, a % b) }
        return a
    }
}

/// One request's aligned prefill: the step `C`, the store boundary `K` and the prompt length `N`, all absolute.
public struct AlignedPrefillPlan: Sendable, Equatable, CustomStringConvertible {
    public let step: Int
    public let boundary: Int?
    public let promptTokenCount: Int

    public init(step: Int, boundary: Int?, promptTokenCount: Int) {
        self.step = max(1, step)
        self.boundary = boundary
        self.promptTokenCount = promptTokenCount
    }

    /// The geometry the rows of a prefill with this plan are produced in.
    public var tag: AlignedPrefillTag { AlignedPrefillTag(step: step, boundary: boundary ?? 0) }

    public var description: String {
        "step=\(step) boundary=\(boundary.map(String.init) ?? "none") prompt=\(promptTokenCount)"
    }
}

/// The geometry a cache's rows were produced in: the prefill step and the producing request's store boundary (0 when
/// it had none). Stored with the state (`metaState`) so every tier can tell which rows a later request may reuse.
public struct AlignedPrefillTag: Sendable, Equatable, CustomStringConvertible {
    public let step: Int
    public let boundary: Int

    public init(step: Int, boundary: Int) {
        self.step = step
        self.boundary = boundary
    }

    /// Rows of every full aligned chunk at or before the producing boundary: reusable by any request whose prompt
    /// shares them.
    public var cleanPrefix: Int { AlignedPrefillGeometry.trailingChunkStart(boundary: boundary, step: step) }

    /// `[schema, "step=C", "boundary=K"]`, appended to a cache's own `metaState`.
    public var metaState: [String] {
        [AlignedPrefillGeometry.schema, "step=\(step)", "boundary=\(boundary)"]
    }

    /// Parses `metaState`; nil for anything else (a malformed or foreign tag is no tag).
    public init?(metaState: [String]) {
        guard metaState.count == 3, metaState[0] == AlignedPrefillGeometry.schema,
            metaState[1].hasPrefix("step="), metaState[2].hasPrefix("boundary="),
            let step = Int(metaState[1].dropFirst("step=".count)),
            let boundary = Int(metaState[2].dropFirst("boundary=".count)),
            step > 0, boundary >= 0
        else { return nil }
        self.init(step: step, boundary: boundary)
    }

    public var description: String { "step=\(step) boundary=\(boundary)" }
}

/// A cache layer of a model that prefills on aligned chunks: it carries the request's plan into `prepare` and the
/// geometry its rows were produced in (persisted with its state).
public protocol AlignedPrefillCache: AnyObject {
    /// The current request's plan, set by the engine before prefill (request-scoped; not persisted, not copied).
    var alignedPrefillPlan: AlignedPrefillPlan? { get set }
    /// The geometry of the rows this cache holds; nil when unknown (not produced by an aligned prefill).
    var alignedPrefillTag: AlignedPrefillTag? { get set }
    /// Rows of every lane this layer keeps (all equal to `offset` when the lanes are in lockstep).
    var alignedPrefillLaneRows: [Int] { get }
}

/// A model whose prompt prefill follows the aligned-prefill rule (its `prepare` reads the plan from its caches).
public protocol AlignedPrefillModel {
    /// The rule's identity when aligned prefill is on for this model, nil when it is off (the environment lever).
    var alignedPrefillSchema: String? { get }
}

/// What a restore alignment decided (test / diagnostics observers).
public struct AlignedPrefillRestoreRecord: Sendable {
    public let matched: Int
    public let reuse: Int
    public let plan: AlignedPrefillPlan
    public let tag: AlignedPrefillTag?
    public let exactGeometry: Bool
    public let offsets: [Int]
    public let laneRows: [Int]
    public let detail: String
    public let reason: String
}

/// What an aligned model's prefill ran (test / diagnostics observers).
public struct AlignedPrefillPrefillRecord: Sendable {
    /// Absolute positions this prefill covered: `[start, end)`; `start` is the cache offset it started from.
    public let start: Int
    public let end: Int
    /// Absolute pieces, the last one returned to the engine for the sampling forward.
    public let pieces: [Range<Int>]
    public let plan: AlignedPrefillPlan?
    /// True when the pieces followed the plan; false for the relative chunking (no plan, or a plan this prefill does
    /// not fit, which is logged).
    public let aligned: Bool
    public let laneRowsBefore: [Int]

    public init(
        start: Int, end: Int, pieces: [Range<Int>], plan: AlignedPrefillPlan?, aligned: Bool, laneRowsBefore: [Int]
    ) {
        self.start = start
        self.end = end
        self.pieces = pieces
        self.plan = plan
        self.aligned = aligned
        self.laneRowsBefore = laneRowsBefore
    }
}

/// Engine-side helpers: the plan of a request, installing it on the caches, and cutting a restored prefix.
public enum AlignedPrefill {
    public final class Hooks: @unchecked Sendable {
        private let lock = NSLock()
        private var restore: ((AlignedPrefillRestoreRecord) -> Void)?
        private var prefill: ((AlignedPrefillPrefillRecord) -> Void)?
        private var logits: ((MLXArray) -> Void)?

        /// Called for every restore alignment (TokenIterator and BatchEngine).
        public var restoreObserver: ((AlignedPrefillRestoreRecord) -> Void)? {
            get { lock.withLock { restore } }
            set { lock.withLock { restore = newValue } }
        }

        /// Called for every prompt prefill of an aligned model.
        public var prefillObserver: ((AlignedPrefillPrefillRecord) -> Void)? {
            get { lock.withLock { prefill } }
            set { lock.withLock { prefill = newValue } }
        }

        /// Called with the output of every forward of an aligned model that can feed the sampler (the last prefill
        /// piece, decode steps, verify rows); never with the non-final prefill pieces, whose logits are not used.
        public var logitsObserver: ((MLXArray) -> Void)? {
            get { lock.withLock { logits } }
            set { lock.withLock { logits = newValue } }
        }

        private var reuseControl: ((AlignedPrefillRestoreRecord) -> Int?)?
        private var omitBoundaryCut = false

        /// CONTROL ONLY (tests that must fail): replaces the reuse a restore alignment computed (nil keeps it), with no
        /// chunk-boundary check, so a continuation can start mid-chunk. Never set in production.
        public var controlReuse: ((AlignedPrefillRestoreRecord) -> Int?)? {
            get { lock.withLock { reuseControl } }
            set { lock.withLock { reuseControl = newValue } }
        }

        /// CONTROL ONLY: an aligned prefill does not cut at the store boundary (the rows still carry it as their
        /// geometry), so an exact repeat continues from a boundary its cold prefill never cut at. Never set in
        /// production.
        public var controlOmitsBoundaryCut: Bool {
            get { lock.withLock { omitBoundaryCut } }
            set { lock.withLock { omitBoundaryCut = newValue } }
        }
    }

    /// Test / diagnostics hooks (nil observers in production).
    public static let hooks = Hooks()

    static func trace(_ message: @autoclosure () -> String) {
        if ProcessInfo.processInfo.environment["VMLX_CACHE_FETCH_TRACE"] == "1" {
            RuntimeDiagnostics.write(Data("[vmlx][cache/aligned-prefill] \(message())\n".utf8))
        }
    }

    /// The aligned layers of a cache list (leaves of `CacheList` included).
    public static func layers(_ cache: [any KVCache]) -> [any AlignedPrefillCache] {
        cache.flatMap { layer -> [any AlignedPrefillCache] in
            if let list = layer as? CacheList { return layers((0 ..< list.count).map { list[$0] }) }
            return (layer as? any AlignedPrefillCache).map { [$0] } ?? []
        }
    }

    /// The cache-salt component of an aligned model at this prefill step (nil when aligned prefill is off): entries
    /// written with another step, or before aligned prefill, are never served to an aligned request.
    public static func saltComponent(model: (any LanguageModel)?, prefillStepSize: Int) -> String? {
        guard let schema = (model as? any AlignedPrefillModel)?.alignedPrefillSchema else { return nil }
        return "alignedPrefill=\(schema):step=\(max(1, prefillStepSize))"
    }

    /// The plan of one request, or nil when the model does not prefill on aligned chunks, the lever is off, or the
    /// cache has no aligned layer (then nothing changes).
    public static func plan(
        model: any LanguageModel, cache: [any KVCache], promptTokenIds: [Int], cachePrefixTokenCounts: [Int],
        generationPromptSuffix: [Int], prefillStepSize: Int
    ) -> AlignedPrefillPlan? {
        guard (model as? any AlignedPrefillModel)?.alignedPrefillSchema != nil, !promptTokenIds.isEmpty,
            !layers(cache).isEmpty
        else { return nil }
        return AlignedPrefillPlan(
            step: prefillStepSize,
            boundary: AlignedPrefillGeometry.storeBoundary(
                promptTokenIds: promptTokenIds, cachePrefixTokenCounts: cachePrefixTokenCounts,
                generationPromptSuffix: generationPromptSuffix),
            promptTokenCount: promptTokenIds.count)
    }

    /// Hand the request's plan to its caches (the model's `prepare` reads it there). nil clears it.
    public static func install(_ plan: AlignedPrefillPlan?, into cache: [any KVCache]) {
        for layer in layers(cache) { layer.alignedPrefillPlan = plan }
    }

    /// The common geometry tag of a cache list's aligned layers, nil when any is missing or they disagree.
    public static func tag(of cache: [any KVCache]) -> AlignedPrefillTag? {
        let tags = layers(cache).map(\.alignedPrefillTag)
        guard let first = tags.first, let tag = first, tags.allSatisfy({ $0 == tag }) else { return nil }
        return tag
    }

    /// Cut a restored prefix of `matchedTokens` rows to what a cold prefill of this request also computes: all of the
    /// request's `K` rows when the stored rows were produced with that same `K` (exact geometry), else the rows below
    /// the last chunk boundary at or before both `K`s. Trims `cache` in place and checks every lane. Returns the rows
    /// to reuse; 0 means none (the caller rebuilds the cache and prefills the whole prompt). Never refuses a request.
    public static func alignRestoredPrefix(
        _ cache: [any KVCache], matchedTokens: Int, plan: AlignedPrefillPlan, detail: String
    ) -> Int {
        let tag = tag(of: cache)
        let offsets = cacheBoundaryLeafOffsets(cache)
        var exact = false
        var reuse = 0
        var reason = "ok"
        if offsets.isEmpty || offsets.contains(where: { $0 != matchedTokens }) {
            reason = "restored offsets \(Array(Set(offsets)).sorted()) != matched \(matchedTokens)"
        } else if let tag, tag.step == plan.step, let boundary = plan.boundary {
            exact = tag.boundary == boundary && matchedTokens >= boundary
            // Rows past the producing boundary were computed in its final piece or decoded: never reusable.
            let usable = exact ? boundary : min(matchedTokens, tag.cleanPrefix)
            reuse = AlignedPrefillGeometry.reuseLimit(
                matched: usable, boundary: boundary, step: plan.step, exactGeometry: exact)
        } else if tag == nil {
            reason = "stored rows carry no aligned geometry"
        } else if plan.boundary == nil {
            reason = "request has no store boundary"
        } else {
            reason = "stored step \(tag!.step) != request step \(plan.step)"
        }
        var controlled = false
        if let control = hooks.controlReuse, offsets.allSatisfy({ $0 == matchedTokens }),
            let forced = control(AlignedPrefillRestoreRecord(
                matched: matchedTokens, reuse: reuse, plan: plan, tag: tag, exactGeometry: exact, offsets: offsets,
                laneRows: layers(cache).flatMap(\.alignedPrefillLaneRows), detail: detail, reason: reason))
        {
            reuse = max(0, min(forced, matchedTokens))
            reason = "control"
            controlled = true
        }
        if reuse > 0, reuse < matchedTokens {
            let trim = matchedTokens - reuse
            for layer in cache where layer.isTrimmable {
                layer.trim(trim)
            }
        }
        let after = cacheBoundaryLeafOffsets(cache)
        let laneRows = layers(cache).flatMap(\.alignedPrefillLaneRows)
        // The invariants every restore must meet before the generator sees it: every layer and lane holds exactly the
        // reused rows, and the reuse is a chunk boundary or this request's own store boundary.
        if reuse > 0,
            after.contains(where: { $0 != reuse }) || laneRows.contains(where: { $0 != reuse })
                || !(controlled || reuse % plan.step == 0 || reuse == plan.boundary)
        {
            reason = "cut to \(reuse) left offsets \(Array(Set(after)).sorted()) lanes \(Array(Set(laneRows)).sorted())"
            reuse = 0
        }
        if reuse == 0 && reason == "ok" { reason = "no aligned prefix" }
        let record = AlignedPrefillRestoreRecord(
            matched: matchedTokens, reuse: reuse, plan: plan, tag: tag, exactGeometry: exact && reuse > 0,
            offsets: reuse > 0 ? after : offsets, laneRows: laneRows, detail: detail, reason: reason)
        hooks.restoreObserver?(record)
        trace(
            "restore detail=\(detail) matched=\(matchedTokens) reuse=\(reuse) exact=\(record.exactGeometry) "
                + "tag=\(tag.map(String.init(describing:)) ?? "nil") plan={\(plan)} reason=\(reason)")
        return reuse
    }
}
