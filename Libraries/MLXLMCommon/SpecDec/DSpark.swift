// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4.1 dSpark speculative decode, run inside `TokenIterator` (like Uno).
//
// Port of the Python reference `jang_tools/dsv41/dspark.py::DSparkRunner.decode_stream` (methods M1
// and M2 of `DECODE_DSPARK_METHODS.md`) and its wall-clock `_Governor`.
//
// State matches ordinary decoding: the cache holds the prompt plus every committed token, and the
// next token `p0` (position c+1, where c+1 == cache offset) is sampled but not yet forwarded. A
// speculative step:
//   1. drafts d1...dW for positions c+2...c+1+W from p0 and the committed hidden rows;
//   2. checkpoints the cache in place, then runs ONE verify forward of [p0, d1...dW];
//   3. accepts the longest prefix where d_{j+1} == argmax(row j);
//   4. full accept: keeps the verify cache as it is (no refeed); the bonus argmax(row W) becomes the
//      next p0. Partial accept with k > 0 (Python `commit_prefix`, M1 step 7): the target recorded
//      the verify's per-token cache inputs, so the cache is restored in place and only the kept
//      tokens [p0, d1...dk] are re-appended (window rows, compressor pooling), with no second
//      forward (each extra forward costs about one more decode step); truth_k becomes the next p0.
//      k == 0 (or a target without capture): restore in place and run one commit forward of
//      [p0, truth_0...].
// Every emitted token is a backbone argmax, so the output is greedy AR's (up to the usual caveat
// that a multi-row forward rounds differently from a one-row decode step, so a near-tie can flip).
// A sampled request (temperature > 0) replaces each argmax by the AR sampler's draw with Gumbel
// noise keyed by the absolute position (``DSparkSampling``, Python `SpecSampler`), and the draft
// sees the same noise, so the output is distributed as AR sampling's. The governor's AR steps draw
// with the same position-keyed noise (`TokenIterator.convertToToken`), so with a fixed seed the
// output does not depend on which steps speculated (a wall-clock decision).
//
// The width W tracks acceptance (EMA, 1...ladderTop). Whether to speculate at all is decided by the
// measured wall-clock cost per emitted token of speculative steps against AR steps
// (``DSparkGovernor``): never a fixed depth, never a permanent latch. The ladder top is the widest
// draft the session can produce: the head's block for DeepSeek-V4.1 (its stages draft exactly that
// many positions), twice the trained block for a MiniMax-M3 block drafter (Python's `ladder_top`: a
// round past the trained block runs a longer block of mask rows). The trained block is only where
// the width starts; an env lever (`VMLX_DSPARK_WIDTH`, MiniMax) may pin it for diagnostics.

import Foundation
import MLX

// MARK: - Model contract

/// A target that carries a dSpark draft head (DeepSeek-V4.1 / MiniMax-M3 `dspark/`).
public protocol DSparkSpeculativeModel: AnyObject {
    /// Draft block size of the attached head (the trained block: where the width starts); nil while no
    /// head is attached.
    var dsparkBlockSize: Int? { get }
    /// The widest draft the controller may choose (default: the block size).
    var dsparkLadderTop: Int? { get }
    /// A width pinned for every step by a diagnostic env lever; nil (the default) lets the controller choose.
    var dsparkPinnedWidth: Int? { get }
    /// Load the head from `directory` once (later calls are no-ops). Returns false when this model
    /// cannot use a dSpark head (for example DSV4-0731).
    func attachDSparkDraft(directory: URL) throws -> Bool
    /// A drafting session with its own draft windows, for one request. nil without a head.
    func makeDSparkDraftSession() -> (any DSparkDraftSession)?
    /// Why `cache` cannot be speculated on (nil when it can).
    func dsparkCacheUnsupportedReason(_ cache: [KVCache]) -> String?
}

extension DSparkSpeculativeModel {
    public var dsparkLadderTop: Int? { dsparkBlockSize }
    public var dsparkPinnedWidth: Int? { nil }
}

/// Per-request drafter state (the per-stage attention windows). Not shared between requests.
public protocol DSparkDraftSession: AnyObject {
    /// Draft the tokens of positions c+2 ... c+1+blockSize. `seed` is the backbone's token at c+1
    /// (pending, not in the cache); the cache offset is c+1. The windows are brought up to date from
    /// the committed hidden rows the cache carries. Returns `blockSize` token ids (one host sync).
    func draft(seed: Int, cache: [KVCache]) -> [Int]

    /// Coupled sampled drafting: as `draft(seed:cache:)`, but step i proposes
    /// argmax(score_i * inverseTemperature + noise[i]), with `noise` `[blockSize, vocab]` the Gumbel
    /// rows the target will sample positions c+2 ... c+1+blockSize with, so a draft close to the
    /// target proposes the target's own sample instead of its argmax.
    func draft(seed: Int, cache: [KVCache], noise: MLXArray, inverseTemperature: Float) -> [Int]
    /// The draft head's vocabulary (the width of its noise rows); nil drafts sampled requests
    /// uncoupled (still exact: a draft is only kept when it equals the target's sample).
    var vocabularySize: Int? { get }

    /// Ask the target to record the next forward's per-token cache inputs (the verify forward).
    /// Returns false when the target cannot commit a verified prefix in place.
    func beginVerifyCapture(cache: [KVCache]) -> Bool
    /// After a captured verify forward of n tokens that followed `checkpoint`: leave the cache
    /// holding exactly the first `keep` (1...n) of them, as a forward of only those tokens would,
    /// without running another forward and without replacing any cache object. Returns false
    /// (cache untouched) when it cannot; the caller then restores and refeeds.
    func commitVerifiedPrefix(
        cache: [KVCache], checkpoint: SpeculativeCacheListCheckpoint, keep: Int) -> Bool
    /// Drop whatever the capture still holds.
    func endVerifyCapture(cache: [KVCache])

    /// Draft `count` tokens (positions c+2 ... c+1+count). A block drafter runs a block of
    /// max(trained block, count) rows; the default drafts the head's block and the caller takes a prefix.
    func draft(seed: Int, cache: [KVCache], count: Int) -> [Int]
    /// Coupled sampled drafting of `count` tokens; `noise` holds at least `count` rows.
    func draft(seed: Int, cache: [KVCache], count: Int, noise: MLXArray, inverseTemperature: Float) -> [Int]
    /// True when ``commitVerifiedPrefix(cache:checkpoint:keep:)`` is as cheap for a zero-accept step
    /// (keep = 1) as for any other, so such a step commits in place instead of a second forward.
    var commitsAnyPrefix: Bool { get }
}

extension DSparkDraftSession {
    public func draft(seed: Int, cache: [KVCache], noise: MLXArray, inverseTemperature: Float) -> [Int] {
        draft(seed: seed, cache: cache)
    }
    public var vocabularySize: Int? { nil }
    public func beginVerifyCapture(cache: [KVCache]) -> Bool { false }
    public func commitVerifiedPrefix(
        cache: [KVCache], checkpoint: SpeculativeCacheListCheckpoint, keep: Int
    ) -> Bool { false }
    public func endVerifyCapture(cache: [KVCache]) {}
    public func draft(seed: Int, cache: [KVCache], count: Int) -> [Int] {
        draft(seed: seed, cache: cache)
    }
    public func draft(
        seed: Int, cache: [KVCache], count: Int, noise: MLXArray, inverseTemperature: Float
    ) -> [Int] {
        draft(seed: seed, cache: cache, noise: noise, inverseTemperature: inverseTemperature)
    }
    public var commitsAnyPrefix: Bool { false }
}

// MARK: - Environment

public enum DSparkEnvironment {
    /// `VMLX_DSV41_DSPARK=0|off|false|no` keeps dSpark off for every request. The variable can only
    /// turn dSpark off: it never enables it over Speculative Decoding Off.
    public static func disabled(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        guard let raw = environment["VMLX_DSV41_DSPARK"]?.lowercased() else { return false }
        return ["0", "off", "false", "no"].contains(raw)
    }

    /// `VMLX_DSPARK=0|off|false|no` (the vMLX Python runtime's lever) keeps MiniMax-M3 DSpark off. Like
    /// `VMLX_DSV41_DSPARK`, it can only turn DSpark off.
    public static func miniMaxDisabled(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        guard let raw = environment["VMLX_DSPARK"]?.lowercased() else { return false }
        return ["0", "off", "false", "no"].contains(raw)
    }

    /// `VMLX_DSV41_DSPARK_SAMPLED=0|off|false|no`: sampled requests (temperature > 0) decode AR
    /// instead of with sampled speculation (``DSparkSampling``). Greedy requests are unaffected.
    public static func sampledDisabled(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        guard let raw = environment["VMLX_DSV41_DSPARK_SAMPLED"]?.lowercased() else { return false }
        return ["0", "off", "false", "no"].contains(raw)
    }

    /// `VMLX_DSV41_DSPARK_STATS=1`: one NSLog line per request (spec/AR steps, acceptance, ms/token).
    public static let statsEnabled =
        ProcessInfo.processInfo.environment["VMLX_DSV41_DSPARK_STATS"] == "1"
}

// MARK: - Sampled speculation

/// Exact sampled speculation for dSpark (Python `SpecSampler`, temperature > 0).
///
/// The token at each target position is argmax(f(logits) + g) with g i.i.d. Gumbel noise, a draw
/// from the distribution this request's AR sampler draws from: f is that sampler's transform
/// (``GenerateParameters/sampler()``: ``TopPSampler`` log-probabilities at the temperature with
/// top_p, min_p, top_k applied, or ``CategoricalSampler`` logits / T). The noise is keyed by the
/// request seed and the ABSOLUTE position, so a position whose row was computed in one step and is
/// decided in a later one replays the same sample, and noise that decided nothing committed stays
/// independent of everything that was (fresh noise per step is biased: a re-sampled rejected
/// position then follows the target conditioned on the draft's rejection). The draft sees the same
/// noise (coupled drafting), so it proposes the target's sample whenever its own distribution is
/// close, instead of its argmax.
public struct DSparkSampling {
    public let temperature: Float
    public let seed: UInt64
    /// The AR sampler's filters (nil: plain categorical at the temperature).
    let filters: TopPSampler?

    public init?(temperature: Float, topP: Float = 1, topK: Int = 0, minP: Float = 0, seed: UInt64?) {
        guard temperature > 0, temperature.isFinite else { return nil }
        self.temperature = temperature
        self.seed = seed ?? UInt64.random(in: 0 ... UInt64(UInt32.max))
        let filtered = (topP > 0 && topP < 1) || topK > 0 || minP > 0
        filters = filtered
            ? TopPSampler(temperature: temperature, topP: topP, topK: topK, minP: minP) : nil
    }

    /// nil for a greedy request.
    public init?(parameters: GenerateParameters) {
        self.init(
            temperature: parameters.temperature, topP: parameters.topP, topK: parameters.topK,
            minP: parameters.minP, seed: parameters.randomSeed)
    }

    func key(position: Int) -> MLXArray {
        MLXRandom.key(
            (seed &* 2_654_435_761 &+ UInt64(position) &* 40503 &+ 1) & 0x7FFF_FFFF_FFFF_FFFF)
    }

    /// Gumbel rows `[count, vocab]` f32 for absolute positions `first ..< first + count`.
    public func noise(first: Int, count: Int, vocab: Int) -> MLXArray {
        stacked((0 ..< count).map { MLXRandom.gumbel([vocab], key: key(position: first + $0)) })
    }

    /// The AR sampler's draw for each row of `logits` `[n, vocab]` with `noise` `[n, vocab]`:
    /// sampled ids `[n]` int32.
    public func pick(_ logits: MLXArray, noise: MLXArray) -> MLXArray {
        let z = filters?.filteredLogprobs(logits) ?? logits.asType(.float32) * (1 / temperature)
        return argMax(z + noise, axis: -1).asType(.int32)
    }

    /// The token at absolute `position` from its single logits row (`[vocab]` or `[1, vocab]`): id `[1]`
    /// int32. ``pick(_:noise:)`` with that position's noise row, so a token decided by an AR step equals the
    /// one a verify row would give it. While dSpark sampling is active every sampled token after the
    /// prompt's comes from here or from a verify row, so the stream depends only on (seed, prompt), never
    /// on which steps the wall-clock governor made speculative.
    public func sample(_ row: MLXArray, position: Int) -> MLXArray {
        let row = row.reshaped(1, -1)
        return pick(row, noise: noise(first: position, count: 1, vocab: row.dim(-1)))
    }
}

// MARK: - Wall-clock governor

/// Port of the Python `_Governor`. Keeps, for speculative steps and for AR steps, an EMA of step
/// seconds and an EMA of tokens EMITTED per step, each step measured from its start to the next step's
/// start (every sync, draft, verify and commit forward included, and the consumer's time between
/// tokens, which is the same per token for both modes). The cost per token is their ratio: a mean of
/// per-step ratios over-weights the one-token steps of a missed draft (a 6-token and a 1-token step of
/// 50 ms each average 29 ms/token, not the 14 ms/token they cost), and one such miss flipped DeepSeek-V4.1
/// code generation into AR stretches once AR got faster (osaurus, JANG_2K: 113 of 150 steps AR with
/// speculation at 14.5 ms/token against AR at 20.8).
///
/// - speculate while spec s/token < AR s/token; otherwise take AR steps;
/// - in AR mode, one speculative probe every `probeEvery` AR tokens (a probe after an AR stretch
///   replaces the stale spec estimate, so one good probe is enough to switch back);
/// - in spec mode, one AR step every `arProbeEvery` spec steps re-measures AR (its cost grows with
///   context);
/// - an unmeasured AR cost is measured after the first `warmupSpec` spec steps.
/// There is no terminal state: both estimates keep being refreshed.
public struct DSparkGovernor: Sendable, Equatable {
    public enum Mode: Sendable, Equatable { case spec, ar }

    public let probeEvery: Int
    public let arProbeEvery: Int
    public let alpha: Double
    public let warmupSpec: Int

    /// EMAs of one mode's seconds per step and tokens emitted per step.
    private struct Steps: Sendable, Equatable {
        var seconds: Double, tokens: Double
        var perToken: Double { seconds / tokens }
    }

    private var specSteps: Steps?
    private var arSteps: Steps?
    public private(set) var arMode = false
    private var sinceProbe = 0
    private var sinceAR = 0
    private var specStale = false
    private var nSpec = 0

    public init(probeEvery: Int = 16, arProbeEvery: Int = 64, alpha: Double = 0.3, warmupSpec: Int = 2) {
        self.probeEvery = Swift.max(1, probeEvery)
        self.arProbeEvery = Swift.max(1, arProbeEvery)
        self.alpha = alpha
        self.warmupSpec = warmupSpec
    }

    /// Seconds per emitted token of speculative steps (nil until measured).
    public var spec: Double? { specSteps?.perToken }
    /// Seconds per emitted token of AR steps (nil until measured).
    public var ar: Double? { arSteps?.perToken }

    private func ema(_ old: Steps?, _ seconds: Double, _ tokens: Double, replace: Bool = false)
        -> Steps
    {
        guard let old, !replace else { return Steps(seconds: seconds, tokens: tokens) }
        return Steps(
            seconds: (1 - alpha) * old.seconds + alpha * seconds,
            tokens: (1 - alpha) * old.tokens + alpha * tokens)
    }

    public mutating func record(_ mode: Mode, seconds: Double, emitted: Int) {
        let tokens = Double(Swift.max(1, emitted))
        switch mode {
        case .spec:
            specSteps = ema(specSteps, seconds, tokens, replace: specStale)
            specStale = false
            nSpec += 1
            sinceAR += 1
        case .ar:
            arSteps = ema(arSteps, seconds, tokens)
            sinceAR = 0
            if arMode { sinceProbe += 1 }
        }
        if let spec, let ar {
            let slower = spec >= ar
            if slower && !arMode {
                arMode = true
                sinceProbe = 0
            } else if !slower && arMode {
                arMode = false
            }
        }
    }

    public mutating func choose() -> Mode {
        if arMode {
            if sinceProbe >= probeEvery {
                sinceProbe = 0
                specStale = true
                return .spec  // probe
            }
            return .ar
        }
        if ar == nil && nSpec >= warmupSpec { return .ar }  // first AR measurement
        if ar != nil && sinceAR >= arProbeEvery { return .ar }  // refresh the AR estimate
        return .spec
    }
}

// MARK: - Stats

public struct DSparkGenerationStats: Sendable, Equatable {
    public var steps = 0
    public var specSteps = 0
    public var arSteps = 0
    /// Draft tokens verified (sum of W).
    public var drafted = 0
    /// Draft tokens accepted (sum of k).
    public var accepted = 0
    /// Tokens committed by speculative steps (AR steps commit one each).
    public var specEmitted = 0
    public var fullAccepts = 0
    /// Partial accepts committed in place (no second forward).
    public var prefixCommits = 0
    public var commitForwards = 0
    public var specSeconds: Double = 0
    public var arSeconds: Double = 0
    public var draftSeconds: Double = 0
    public var verifySeconds: Double = 0
    public var commitSeconds: Double = 0
    public var specSecondsPerToken: Double?
    public var arSecondsPerToken: Double?
    /// Sampled speculation (``DSparkSampling``) rather than greedy.
    public var sampled = false
    /// Verify forwards by width (rows = the pending token + the drafts): the draft depths Adaptive chose.
    public var verifyWidths: [Int: Int] = [:]

    public init() {}

    public var acceptanceRate: Double { drafted > 0 ? Double(accepted) / Double(drafted) : 0 }
    public var emitted: Int { specEmitted + arSteps }

    public var summary: String {
        func ms(_ s: Double?) -> String { s.map { String(format: "%.1f", $0 * 1000) } ?? "-" }
        let specTok = specEmitted > 0 ? specSeconds / Double(specEmitted) : nil
        let arTok = arSteps > 0 ? arSeconds / Double(arSteps) : nil
        let total = specSeconds + arSeconds
        let tps = total > 0 ? Double(emitted) / total : 0
        return
            "\(sampled ? "sampled " : "")steps=\(steps) spec=\(specSteps) ar=\(arSteps) drafted=\(drafted) "
            + "accepted=\(accepted) "
            + String(format: "accept=%.3f ", acceptanceRate)
            + "fullAccept=\(fullAccepts) prefixCommit=\(prefixCommits) commitForwards=\(commitForwards) "
            + "emitted=\(emitted) "
            + String(format: "mean_emit=%.2f ", specSteps > 0 ? Double(specEmitted) / Double(specSteps) : 0)
            + "ms/tok spec=\(ms(specTok)) ar=\(ms(arTok)) "
            + "gov spec=\(ms(specSecondsPerToken)) ar=\(ms(arSecondsPerToken)) "
            + "draft_ms=\(ms(draftSeconds)) verify_ms=\(ms(verifySeconds)) commit_ms=\(ms(commitSeconds)) "
            + String(format: "tok/s=%.1f", tps)
    }
}

// MARK: - Decoder

/// The dSpark decode loop for one request. Owned by `TokenIterator`; drives the target cache it is
/// handed, in place.
public struct DSparkDecoder {
    public enum Step {
        /// Take one ordinary decode step for the seed (governor chose AR, or speculation is not
        /// possible for this step).
        case autoregressive
        /// Committed tokens (the first is the seed) are in the cache; `next` is the new pending token.
        case committed([Int], next: MLXArray)
    }

    public let session: any DSparkDraftSession
    public let blockSize: Int
    /// The widest draft the width may reach (>= 1; the block size unless the target declares more).
    public let ladderTop: Int
    /// A diagnostic pin of every step's width (an env lever, at most `ladderTop`); nil lets acceptance choose.
    public let pinnedWidth: Int?
    public let stopTokenIDs: Set<Int>
    /// Sampled speculation (temperature > 0); nil decodes greedily.
    public let sampling: DSparkSampling?
    public private(set) var governor: DSparkGovernor
    public private(set) var stats = DSparkGenerationStats()
    /// Rolling acceptance (`accept_ema`): the draft width W = clamp(round(ladderTop * ema), 1, ladderTop).
    /// Starts at blockSize / ladderTop, so the first width is the trained block.
    public private(set) var acceptEMA: Double = 1.0
    let clock: () -> Double
    /// The step in flight; `timed` is false when it ran a forward width this process had not run yet.
    private var last: (mode: DSparkGovernor.Mode, emitted: Int, start: Double, timed: Bool)?
    /// Start time of the step ``begin(remaining:)`` opened.
    private var stepStart: Double = 0

    public init(
        session: any DSparkDraftSession, blockSize: Int, stopTokenIDs: Set<Int>,
        sampling: DSparkSampling? = nil,
        governor: DSparkGovernor = DSparkGovernor(),
        ladderTop: Int? = nil, pinnedWidth: Int? = nil,
        clock: @escaping () -> Double = { Date.timeIntervalSinceReferenceDate }
    ) {
        precondition(blockSize >= 1, "dSpark block size must be at least 1")
        self.session = session
        self.blockSize = blockSize
        self.ladderTop = Swift.max(1, ladderTop ?? blockSize)
        self.pinnedWidth = pinnedWidth.map { Swift.max(1, $0) }
        acceptEMA = Swift.min(1, Double(blockSize) / Double(self.ladderTop))
        self.stopTokenIDs = stopTokenIDs
        self.sampling = sampling
        stats.sampled = sampling != nil
        self.governor = governor
        self.clock = clock
    }

    /// Current draft width.
    public var width: Int {
        if let pinnedWidth { return Swift.min(pinnedWidth, ladderTop) }
        return Swift.max(1, Swift.min(ladderTop, Int((Double(ladderTop) * acceptEMA).rounded())))
    }

    /// One step from the pending token `seed`, which the caller has already read back (so the
    /// previous step's work is complete and its time can be closed here).
    ///
    /// - Parameters:
    ///   - remaining: tokens the request may still return, counting `seed`.
    ///   - forward: one target forward of the given token ids at the cache's current offset,
    ///     returning logits `[1, L, vocab]`.
    public mutating func step(
        seed: Int, cache: [KVCache], remaining: Int,
        forward: ([Int]) -> MLXArray
    ) -> Step {
        guard begin(remaining: remaining) else { return .autoregressive }
        return speculate(seed: seed, cache: cache, remaining: remaining, forward: forward)
    }

    /// Start a step: close the previous step's timing and let the governor choose. Returns true when this step
    /// speculates (then call ``speculate(seed:cache:remaining:forward:)`` with the pending token); false when the
    /// caller takes one ordinary decode step. Neither needs the pending token's value, so an AR step never waits for
    /// it: the caller's ordinary step submits the next forward before reading the previous token, as plain decoding
    /// does (reading it here first left the GPU idle while each AR step's graph was built).
    public mutating func begin(remaining: Int) -> Bool {
        stats.steps += 1
        let now = clock()
        closeLastStep(now: now)
        stepStart = now
        // A block needs room for at least the seed plus one draft.
        guard governor.choose() == .spec, remaining >= 2 else {
            stats.arSteps += 1
            last = (.ar, 1, now, !DSparkForwardWidths.firstUse(1))
            return false
        }
        return true
    }

    /// The speculative step ``begin(remaining:)`` chose, from the pending token `seed` (read back by the caller).
    public mutating func speculate(
        seed: Int, cache: [KVCache], remaining: Int,
        forward: ([Int]) -> MLXArray
    ) -> Step {
        let now = stepStart
        // No stop token may be pending.
        guard !stopTokenIDs.contains(seed), let checkpoint = SpeculativeCacheListCheckpoint(cache) else {
            stats.arSteps += 1
            last = (.ar, 1, now, !DSparkForwardWidths.firstUse(1))
            return .autoregressive
        }

        stats.specSteps += 1
        let w = Swift.min(width, remaining - 1)
        // The seed's absolute position (pending: the cache holds everything before it). A sampled
        // step decides positions position+1 ... with the noise rows keyed by those positions.
        let position = cache.map(\.offset).max() ?? 0
        var noise: MLXArray?
        let draftStart = clock()
        let drafts: [Int]
        // Rows the session drafts: at least the head's block (a DSV4.1 head drafts exactly that many),
        // `w` for a block drafter asked for more.
        let span = Swift.max(blockSize, w)
        if let sampling, let vocab = session.vocabularySize {
            let rows = sampling.noise(first: position + 1, count: span + 1, vocab: vocab)
            noise = rows
            drafts = session.draft(
                seed: seed, cache: cache, count: w, noise: rows[..<span],
                inverseTemperature: 1 / sampling.temperature)
        } else {
            drafts = session.draft(seed: seed, cache: cache, count: w)
        }
        stats.draftSeconds += clock() - draftStart
        let proposed = Array(drafts.prefix(w))
        let feed = [seed] + proposed
        stats.drafted += proposed.count
        stats.verifyWidths[feed.count, default: 0] += 1

        // Ask the target to keep this forward's per-token cache inputs, so a partial accept can keep
        // the verified prefix without a second forward.
        let capturing = session.beginVerifyCapture(cache: cache)
        defer { if capturing { session.endVerifyCapture(cache: cache) } }
        let verifyStart = clock()
        let logits = forward(feed)
        let truth: [Int]
        if let sampling {
            // The target's samples for positions position+1 ... position+feed.count.
            let rows = logits[0]
            let vocab = rows.dim(-1)
            let n = noise.flatMap { $0.dim(-1) == vocab ? $0[..<feed.count] : nil }
                ?? sampling.noise(first: position + 1, count: feed.count, vocab: vocab)
            truth = sampling.pick(rows, noise: n).asArray(Int32.self).map(Int.init)
        } else {
            truth = argMax(logits[0], axis: -1).asType(.int32).asArray(Int32.self).map(Int.init)
        }
        stats.verifySeconds += clock() - verifyStart

        var k = 0
        while k < proposed.count, proposed[k] == truth[k] { k += 1 }
        stats.accepted += k
        acceptEMA = 0.7 * acceptEMA + 0.3 * (Double(k) / Double(Swift.max(1, proposed.count)))

        let fullAccept = k == proposed.count
        // Never commit past a stop token or the budget: the cache must end exactly where ordinary
        // decoding's would after returning these tokens.
        func cut(_ tokens: [Int]) -> Int {
            var n = Swift.min(tokens.count, remaining)
            if let stop = tokens.firstIndex(where: { stopTokenIDs.contains($0) }) {
                n = Swift.min(n, stop + 1)
            }
            return n
        }
        var emit: [Int]
        let next: MLXArray
        var commitWidth: Int? = nil
        // Verified tokens of `feed`: all of it on a full accept, else p0 plus the k accepted drafts.
        let verified = fullAccept ? feed.count : k + 1
        let keep = cut(Array(feed.prefix(verified)))
        if fullAccept && keep == feed.count {
            // No-refeed fast path: the verify forward already wrote exact causal state for every
            // emitted position; argmax(row W) is the next p0.
            stats.fullAccepts += 1
            emit = feed
            next = MLXArray([Int32(truth[proposed.count])])
        } else if capturing, k > 0 || fullAccept || session.commitsAnyPrefix,
            session.commitVerifiedPrefix(cache: cache, checkpoint: checkpoint, keep: keep)
        {
            // Partial accept (or a cut block), committed in place (Python `commit_prefix`): the
            // cache now holds exactly the first `keep` verify tokens; truth[keep - 1] is the next p0.
            stats.prefixCommits += 1
            emit = Array(feed.prefix(keep))
            next = MLXArray([Int32(truth[keep - 1])])
        } else {
            // Restore and run one commit forward of exactly the emitted tokens (k == 0 keeps this:
            // it commits two tokens for its second forward).
            emit = fullAccept ? feed : [seed] + proposed.prefix(k) + [truth[k]]
            emit = Array(emit.prefix(cut(emit)))
            checkpoint.restore()
            let commitStart = clock()
            let commit = forward(emit)
            commitWidth = emit.count
            if let sampling {
                // The next pending token is position + emit.count's sample.
                next = sampling.sample(commit[0, -1], position: position + emit.count)
            } else {
                next = argMax(commit[0, -1], axis: -1).asType(.int32).reshaped(1)
            }
            asyncEval(next)
            stats.commitSeconds += clock() - commitStart
            stats.commitForwards += 1
        }
        stats.specEmitted += emit.count
        var newWidth = DSparkForwardWidths.firstUse(feed.count)
        if let commitWidth {
            newWidth = DSparkForwardWidths.firstUse(commitWidth) || newWidth
        }
        last = (.spec, emit.count, now, !newWidth)
        return .committed(emit, next: next)
    }

    /// Close the timing of the step in flight (call when the request ends).
    public mutating func finish() {
        closeLastStep(now: clock())
        last = nil
    }

    private mutating func closeLastStep(now: Double) {
        guard let last else { return }
        let seconds = Swift.max(0, now - last.start)
        if last.timed {
            governor.record(last.mode, seconds: seconds, emitted: last.emitted)
        }
        switch last.mode {
        case .spec: stats.specSeconds += seconds
        case .ar: stats.arSeconds += seconds
        }
        stats.specSecondsPerToken = governor.spec
        stats.arSecondsPerToken = governor.ar
        self.last = nil
    }
}

/// Forward widths (rows per forward) this process has run. A model's first forward at a new width pays
/// one-time costs: compiled decode regions record their trace, and DeepSeek-V4.1's fused decode kernels
/// run a bitwise admission check against the stock path, an eager evaluation per layer (one first
/// verify took ~210 ms instead of ~25). Timing such a step made speculation look slower than AR, so the
/// first reply ran mostly AR (live: code 51.4 tok/s vs 62-110 on later prompts); the governor skips it.
enum DSparkForwardWidths {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var seen = Set<Int>()

    /// True the first time `rows` is reported in this process.
    static func firstUse(_ rows: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return seen.insert(rows).inserted
    }
}
