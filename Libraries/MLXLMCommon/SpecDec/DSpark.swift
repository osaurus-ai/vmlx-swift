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
//      next p0. Partial accept: restores in place and runs one commit forward of exactly the emitted
//      tokens [p0, d1...dk, truth_k].
// Every emitted token is a backbone argmax, so the output is greedy AR's (up to the usual caveat
// that a multi-row forward rounds differently from a one-row decode step at an exact tie).
//
// The width W tracks acceptance (EMA, 1...blockSize). Whether to speculate at all is decided by the
// measured wall-clock cost per emitted token of speculative steps against AR steps
// (``DSparkGovernor``): never a fixed depth, never a permanent latch.

import Foundation
import MLX

// MARK: - Model contract

/// A target that carries a dSpark draft head (DeepSeek-V4.1 `dspark/`).
public protocol DSparkSpeculativeModel: AnyObject {
    /// Draft block size of the attached head; nil while no head is attached.
    var dsparkBlockSize: Int? { get }
    /// Load the head from `directory` once (later calls are no-ops). Returns false when this model
    /// cannot use a dSpark head (for example DSV4-0731).
    func attachDSparkDraft(directory: URL) throws -> Bool
    /// A drafting session with its own draft windows, for one request. nil without a head.
    func makeDSparkDraftSession() -> (any DSparkDraftSession)?
    /// Why `cache` cannot be speculated on (nil when it can).
    func dsparkCacheUnsupportedReason(_ cache: [KVCache]) -> String?
}

/// Per-request drafter state (the per-stage attention windows). Not shared between requests.
public protocol DSparkDraftSession: AnyObject {
    /// Draft the tokens of positions c+2 ... c+1+blockSize. `seed` is the backbone's token at c+1
    /// (pending, not in the cache); the cache offset is c+1. The windows are brought up to date from
    /// the committed hidden rows the cache carries. Returns `blockSize` token ids (one host sync).
    func draft(seed: Int, cache: [KVCache]) -> [Int]
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

    /// `VMLX_DSV41_DSPARK_STATS=1`: one NSLog line per request (spec/AR steps, acceptance, ms/token).
    public static let statsEnabled =
        ProcessInfo.processInfo.environment["VMLX_DSV41_DSPARK_STATS"] == "1"
}

// MARK: - Wall-clock governor

/// Port of the Python `_Governor`. Keeps an EMA of seconds per EMITTED token for speculative steps
/// and for AR steps, each measured from one step's start to the next step's start (every sync,
/// draft, verify and commit forward included, and the consumer's time between tokens, which is the
/// same per token for both modes).
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

    public private(set) var spec: Double?
    public private(set) var ar: Double?
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

    private func ema(_ old: Double?, _ x: Double, replace: Bool = false) -> Double {
        guard let old, !replace else { return x }
        return (1 - alpha) * old + alpha * x
    }

    public mutating func record(_ mode: Mode, seconds: Double, emitted: Int) {
        let per = seconds / Double(Swift.max(1, emitted))
        switch mode {
        case .spec:
            spec = ema(spec, per, replace: specStale)
            specStale = false
            nSpec += 1
            sinceAR += 1
        case .ar:
            ar = ema(ar, per)
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
    public var commitForwards = 0
    public var specSeconds: Double = 0
    public var arSeconds: Double = 0
    public var draftSeconds: Double = 0
    public var verifySeconds: Double = 0
    public var commitSeconds: Double = 0
    public var specSecondsPerToken: Double?
    public var arSecondsPerToken: Double?

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
            "steps=\(steps) spec=\(specSteps) ar=\(arSteps) drafted=\(drafted) accepted=\(accepted) "
            + String(format: "accept=%.3f ", acceptanceRate)
            + "fullAccept=\(fullAccepts) commitForwards=\(commitForwards) emitted=\(emitted) "
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
    public let stopTokenIDs: Set<Int>
    public private(set) var governor: DSparkGovernor
    public private(set) var stats = DSparkGenerationStats()
    /// Rolling acceptance (`accept_ema`): the draft width W = clamp(round(blockSize * ema), 1, bs).
    public private(set) var acceptEMA: Double = 1.0
    let clock: () -> Double
    private var last: (mode: DSparkGovernor.Mode, emitted: Int, start: Double)?

    public init(
        session: any DSparkDraftSession, blockSize: Int, stopTokenIDs: Set<Int>,
        governor: DSparkGovernor = DSparkGovernor(),
        clock: @escaping () -> Double = { Date.timeIntervalSinceReferenceDate }
    ) {
        precondition(blockSize >= 1, "dSpark block size must be at least 1")
        self.session = session
        self.blockSize = blockSize
        self.stopTokenIDs = stopTokenIDs
        self.governor = governor
        self.clock = clock
    }

    /// Current draft width.
    public var width: Int {
        Swift.max(1, Swift.min(blockSize, Int((Double(blockSize) * acceptEMA).rounded())))
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
        stats.steps += 1
        let now = clock()
        closeLastStep(now: now)
        let mode = governor.choose()
        // A block needs room for at least the seed plus one draft, and no stop token may be pending.
        guard mode == .spec, remaining >= 2, !stopTokenIDs.contains(seed),
            let checkpoint = SpeculativeCacheListCheckpoint(cache)
        else {
            stats.arSteps += 1
            last = (.ar, 1, now)
            return .autoregressive
        }

        stats.specSteps += 1
        let w = Swift.min(width, remaining - 1)
        let draftStart = clock()
        let drafts = session.draft(seed: seed, cache: cache)
        stats.draftSeconds += clock() - draftStart
        let proposed = Array(drafts.prefix(w))
        let feed = [seed] + proposed
        stats.drafted += proposed.count

        let verifyStart = clock()
        let logits = forward(feed)
        let truth = argMax(logits[0], axis: -1).asType(.int32).asArray(Int32.self).map(Int.init)
        stats.verifySeconds += clock() - verifyStart

        var k = 0
        while k < proposed.count, proposed[k] == truth[k] { k += 1 }
        stats.accepted += k
        acceptEMA = 0.7 * acceptEMA + 0.3 * (Double(k) / Double(Swift.max(1, proposed.count)))

        let fullAccept = k == proposed.count
        var emit = fullAccept ? feed : [seed] + proposed.prefix(k) + [truth[k]]
        // Never commit past a stop token or the budget: the cache must end exactly where ordinary
        // decoding's would after returning these tokens.
        var cut = Swift.min(emit.count, remaining)
        if let stop = emit.firstIndex(where: { stopTokenIDs.contains($0) }) {
            cut = Swift.min(cut, stop + 1)
        }
        let next: MLXArray
        if fullAccept && cut == emit.count {
            // The verify forward already wrote exact causal state for every emitted position.
            stats.fullAccepts += 1
            next = MLXArray([Int32(truth[proposed.count])])
        } else {
            emit = Array(emit.prefix(cut))
            checkpoint.restore()
            let commitStart = clock()
            let commit = forward(emit)
            next = argMax(commit[0, -1], axis: -1).asType(.int32).reshaped(1)
            asyncEval(next)
            stats.commitSeconds += clock() - commitStart
            stats.commitForwards += 1
        }
        stats.specEmitted += emit.count
        last = (.spec, emit.count, now)
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
        governor.record(last.mode, seconds: seconds, emitted: last.emitted)
        switch last.mode {
        case .spec: stats.specSeconds += seconds
        case .ar: stats.arSeconds += seconds
        }
        stats.specSecondsPerToken = governor.spec
        stats.arSecondsPerToken = governor.ar
        self.last = nil
    }
}
