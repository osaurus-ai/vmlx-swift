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
//
// Turn cuts (v2, the port of vMLX Python's structural rule, 05-PY-STRUCTURAL-CACHE.md): when the model resolves its
// chat template's message markers through the serving tokenizer (`AlignedPrefillTurnMarkers`), a text prefill is ALSO
// cut at every TURN CUT: the position right after an end-of-message token that closes a message other than an
// assistant turn (system, developer, user, tool). The cut is decided by the tokens before it (lookbehind), so the cut
// set below any row r depends only on tokens [0, r): any stored prefix whose first r tokens a request shares was
// computed in exactly the chunks that request's cold prefill runs below r, whenever r is one of its cuts and r is at or
// below the producer's K. A tool loop's next request then reuses the whole previous prompt up to its K (the end of
// the last tool / user message) instead of its last C multiple. Assistant ends are not cuts (a tool call's increment
// [K_prev, K) stays one piece). `VMLX_M3_STRUCTURAL_CUTS=0` keeps the v1 chunk-only rule; v1 and v2 entries never
// serve each other (schema in the salt and the tag).

/// The aligned-prefill rule: pure functions of positions, no model state.
public enum AlignedPrefillGeometry {
    /// Identity of the rule. Part of the cache salt and of every stored state's tag.
    public static let schema = "aligned-prefill-v1"
    /// The rule with turn cuts (v2).
    public static let turnsSchema = "aligned-prefill-v2-turns"
    /// `VMLX_M3_STRUCTURAL_CUTS=0` (or false / off / no): the v1 chunk-only rule even when markers resolve.
    public static let structuralEnvironmentKey = "VMLX_M3_STRUCTURAL_CUTS"

    public static func structuralCutsEnabled(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        let value = (environment[structuralEnvironmentKey] ?? "1").trimmingCharacters(in: .whitespaces).lowercased()
        return !["0", "false", "off", "no"].contains(value)
    }

    /// Every turn cut `c` in `(0, end]`: `tokens[c - 1]` is `markers.eos` and the message it closes (the one opened by
    /// the last `markers.bos` before it) is not an assistant turn; an end-of-message with no open message counts as a
    /// non-assistant end. Python `prefill_geometry.turn_cuts`, bit for bit.
    public static func turnCuts(_ tokens: [Int], markers: AlignedPrefillTurnMarkers?, end: Int? = nil) -> [Int] {
        guard let markers else { return [] }
        let n = end.map { max(0, min($0, tokens.count)) } ?? tokens.count
        let width = markers.assistant.count
        var cuts: [Int] = []
        var assistantOpen = false
        var i = 0
        while i < n {
            let t = tokens[i]
            if t == markers.bos {
                let lo = i + 1, hi = min(tokens.count, i + 1 + width)
                assistantOpen = hi - lo == width && Array(tokens[lo ..< hi]) == markers.assistant
            } else if t == markers.eos {
                if !assistantOpen { cuts.append(i + 1) }
                assistantOpen = false
            }
            i += 1
        }
        return cuts
    }
    /// `VMLX_M3_ALIGNED_PREFIX=0` (or false / off / no) turns aligned prefill off.
    public static let environmentKey = "VMLX_M3_ALIGNED_PREFIX"

    public static func isEnabled(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        let value = (environment[environmentKey] ?? "1").trimmingCharacters(in: .whitespaces).lowercased()
        return !["0", "false", "off", "no"].contains(value)
    }

    /// Absolute `[a, b)` pieces that prefill `[start, end)`: cut at every multiple of `step` and every turn cut in
    /// `turnCuts` (neither for an atomic media prefill), and at `boundary`, when they fall strictly inside.
    public static func pieces(
        start: Int, end: Int, step: Int, boundary: Int?, atomic: Bool = false, turnCuts: [Int] = []
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
            for c in turnCuts where start < c && c < end { cuts.insert(c) }
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
    /// With `turnCuts` (this request's), the largest of those at or below the limit also qualifies (only when a
    /// multiple of `granule` on a block-granular tier). Python `reuse_limit`.
    public static func reuseLimit(
        matched: Int, boundary: Int, step: Int, exactGeometry: Bool, atomic: Bool = false, granule: Int = 1,
        turnCuts: [Int] = []
    ) -> Int {
        guard matched > 0, boundary > 0 else { return 0 }
        if exactGeometry && matched == boundary { return boundary }
        if atomic { return 0 }
        var unit = max(1, step)
        if granule > 1 { unit = unit / gcd(unit, granule) * granule }
        let limit = min(matched, boundary)
        var best = (limit / unit) * unit
        for c in turnCuts where best < c && c <= limit && (granule <= 1 || c % granule == 0) { best = c }
        return best
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
    /// The request's turn cuts (v2), all at or below `boundary`; nil = the v1 chunk-only rule (no markers).
    public let turnCuts: [Int]?

    public init(step: Int, boundary: Int?, promptTokenCount: Int, turnCuts: [Int]? = nil) {
        self.step = max(1, step)
        self.boundary = boundary
        self.promptTokenCount = promptTokenCount
        self.turnCuts = turnCuts
    }

    /// The geometry the rows of a prefill with this plan are produced in.
    public var tag: AlignedPrefillTag {
        AlignedPrefillTag(step: step, boundary: boundary ?? 0, turns: turnCuts != nil)
    }

    public var description: String {
        "step=\(step) boundary=\(boundary.map(String.init) ?? "none") prompt=\(promptTokenCount)"
            + (turnCuts.map { " turns=\($0.count)" } ?? "")
    }
}

/// The geometry a cache's rows were produced in: the prefill step and the producing request's store boundary (0 when
/// it had none). Stored with the state (`metaState`) so every tier can tell which rows a later request may reuse.
public struct AlignedPrefillTag: Sendable, Equatable, CustomStringConvertible {
    public let step: Int
    public let boundary: Int
    /// Produced by a prefill that also cut at its turn cuts (v2).
    public let turns: Bool

    public init(step: Int, boundary: Int, turns: Bool = false) {
        self.step = step
        self.boundary = boundary
        self.turns = turns
    }

    /// Rows of every full aligned chunk at or before the producing boundary: reusable by any request whose prompt
    /// shares them.
    public var cleanPrefix: Int { AlignedPrefillGeometry.trailingChunkStart(boundary: boundary, step: step) }

    /// `[schema, "step=C", "boundary=K"]`, appended to a cache's own `metaState` (schema v1 or v2-turns).
    public var metaState: [String] {
        [turns ? AlignedPrefillGeometry.turnsSchema : AlignedPrefillGeometry.schema, "step=\(step)", "boundary=\(boundary)"]
    }

    /// Parses `metaState`; nil for anything else (a malformed or foreign tag is no tag).
    public init?(metaState: [String]) {
        guard metaState.count == 3,
            metaState[0] == AlignedPrefillGeometry.schema || metaState[0] == AlignedPrefillGeometry.turnsSchema,
            metaState[1].hasPrefix("step="), metaState[2].hasPrefix("boundary="),
            let step = Int(metaState[1].dropFirst("step=".count)),
            let boundary = Int(metaState[2].dropFirst("boundary=".count)),
            step > 0, boundary >= 0
        else { return nil }
        self.init(step: step, boundary: boundary, turns: metaState[0] == AlignedPrefillGeometry.turnsSchema)
    }

    public var description: String { "step=\(step) boundary=\(boundary)" + (turns ? " turns" : "") }
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
    /// The chat template's message markers when the rule cuts at turns (v2), nil for the chunk-only rule.
    var alignedPrefillTurnMarkers: AlignedPrefillTurnMarkers? { get }
}

extension AlignedPrefillModel {
    public var alignedPrefillTurnMarkers: AlignedPrefillTurnMarkers? { nil }
}

/// An aligned model whose turn markers come from the serving tokenizer: `ModelContext` hands it the tokenizer it was
/// loaded with (every load path builds one), so the salt and every plan agree on the markers.
public protocol AlignedPrefillTurnMarkerResolving: AnyObject {
    func resolveAlignedPrefillTurnMarkers(tokenizer: any Tokenizer)
}

/// Token ids of a chat template's message structure: `bos` starts a message (followed by its role tokens), `eos` ends
/// one; `assistant` = the role tokens that follow `bos` in an assistant turn. Python `prefill_geometry.TurnMarkers`.
public struct AlignedPrefillTurnMarkers: Sendable, Equatable, CustomStringConvertible {
    public let bos: Int
    public let eos: Int
    public let assistant: [Int]

    public init(bos: Int, eos: Int, assistant: [Int]) {
        self.bos = bos
        self.eos = eos
        self.assistant = assistant
    }

    /// `<bos>-<eos>-<role ids>`: part of the cache salt, so entries cut at other markers are never served.
    public var signature: String { "\(bos)-\(eos)-" + assistant.map(String.init).joined(separator: ".") }
    public var description: String { signature }

    /// Resolve through the serving tokenizer. Accepted only when the tokenizer encodes the template's joint exactly as
    /// the template renders it: `eosText + "\n" + bosText + assistantRoleText` as `[eos, <newline ids>, bos, <role
    /// ids>]` with single-token `bos` / `eos` (Python `turn_markers_from_tokenizer`). Anything else: nil (no cuts).
    public static func resolve(
        tokenizer: any Tokenizer, bosText: String, eosText: String, assistantRoleText: String
    ) -> AlignedPrefillTurnMarkers? {
        let bos = tokenizer.encode(text: bosText, addSpecialTokens: false)
        let eos = tokenizer.encode(text: eosText, addSpecialTokens: false)
        let role = tokenizer.encode(text: assistantRoleText, addSpecialTokens: false)
        let newline = tokenizer.encode(text: "\n", addSpecialTokens: false)
        let joint = tokenizer.encode(text: eosText + "\n" + bosText + assistantRoleText, addSpecialTokens: false)
        guard bos.count == 1, eos.count == 1, !role.isEmpty, !newline.isEmpty, bos[0] != eos[0],
            joint == eos + newline + bos + role
        else { return nil }
        return AlignedPrefillTurnMarkers(bos: bos[0], eos: eos[0], assistant: role)
    }
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
        private var omitTurnCuts = false

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

        /// CONTROL ONLY: an aligned prefill does not cut at the plan's turn cuts (the rows still carry the v2 tag), so a
        /// later request reuses rows at a turn cut its producer never cut at. Never set in production.
        public var controlOmitsTurnCuts: Bool {
            get { lock.withLock { omitTurnCuts } }
            set { lock.withLock { omitTurnCuts = newValue } }
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
        guard let aligned = model as? any AlignedPrefillModel, aligned.alignedPrefillSchema != nil,
            !promptTokenIds.isEmpty, !layers(cache).isEmpty
        else { return nil }
        let boundary = AlignedPrefillGeometry.storeBoundary(
            promptTokenIds: promptTokenIds, cachePrefixTokenCounts: cachePrefixTokenCounts,
            generationPromptSuffix: generationPromptSuffix)
        let cuts = aligned.alignedPrefillTurnMarkers.map {
            AlignedPrefillGeometry.turnCuts(promptTokenIds, markers: $0, end: boundary ?? 0)
        }
        return AlignedPrefillPlan(
            step: prefillStepSize, boundary: boundary, promptTokenCount: promptTokenIds.count, turnCuts: cuts)
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
        } else if let tag, tag.step == plan.step, let boundary = plan.boundary, tag.turns == (plan.turnCuts != nil) {
            exact = tag.boundary == boundary && matchedTokens >= boundary
            // Rows past the producing boundary were computed in its final piece or decoded: never reusable. Under the
            // turn rule every row up to the producer's K was cut at the producer's cuts, which below any shared row r
            // are this request's own (lookbehind); without it only whole chunks below the producer's K qualify.
            let usable = exact ? boundary : min(matchedTokens, tag.turns ? tag.boundary : tag.cleanPrefix)
            reuse = AlignedPrefillGeometry.reuseLimit(
                matched: usable, boundary: boundary, step: plan.step, exactGeometry: exact,
                turnCuts: plan.turnCuts ?? [])
        } else if let tag, tag.turns != (plan.turnCuts != nil) {
            reason = "stored rule \(tag.turns ? "v2-turns" : "v1") != request rule \(plan.turnCuts != nil ? "v2-turns" : "v1")"
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
                || !(controlled || reuse % plan.step == 0 || reuse == plan.boundary
                    || (plan.turnCuts ?? []).contains(reuse))
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
