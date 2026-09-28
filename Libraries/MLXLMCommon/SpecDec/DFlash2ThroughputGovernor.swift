// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Drafting has to pay for itself, measured, on the turn at hand.
//
// A DFlash 2 cycle costs a drafter forward plus a multi-row verify — on a
// 4B quantized target about 2.2 plain decode steps. It pays only while the
// target accepts more than that many tokens per verify, and acceptance is a
// property of the content: code and structured text accept well, sampled
// free prose and long reasoning do not. Measured live on Spark2.5 4B
// (bundle sampler T=1, reasoning on): drafting decoded at 84-98 tok/s
// against 109-111 tok/s plain on the same turns.
//
// Both paths emit exactly the target's tokens, so choosing between them is
// purely a speed decision. The governor makes it from wall-clock throughput
// measured on this turn — tokens emitted per second between the iterator's
// own cycles, host work included — never from a predicted cost:
//
//   - drafting runs in windows of `windowCycles` verify cycles;
//   - the first window is followed by a short plain stretch that measures
//     plain throughput at this context length;
//   - drafting is set aside after one window clearly slower than plain
//     (`clearLoss`) or two consecutive windows slower at all
//     (`tolerance`): one window of 12 cycles is noisy under sampling, and
//     a pause costs a stretch of plain steps, which inside this iterator
//     run ~9% slower than the plain decoder (they also record the drafter's
//     context);
//   - a pause hands the next `backoff` tokens to plain decoding, then
//     drafting is tried again, doubling the backoff while it keeps losing
//     and resetting it as soon as it wins.
//
// It never refuses and never changes what is emitted.

import Foundation

struct DFlash2ThroughputGovernor: Sendable, Equatable {
    enum Mode: Sendable, Equatable {
        case drafting
        case plain
    }

    /// Verify cycles per measured drafting window.
    static let windowCycles = 12
    /// Plain tokens decoded to measure plain throughput.
    static let calibrationTokens = 16
    static let initialBackoff = 64
    static let maxBackoff = 512
    /// A window below this fraction of plain counts as losing.
    static let tolerance = 0.97
    /// A single window below this fraction of plain is a clear loss.
    static let clearLoss = 0.85

    private(set) var mode: Mode = .drafting
    /// Plain tokens per second measured on this turn.
    private(set) var plainRate: Double?
    /// Tokens per second of the last completed drafting window.
    private(set) var draftRate: Double?
    /// Times drafting was set aside for plain decoding.
    private(set) var pauses = 0
    /// Tokens decoded plainly because drafting was not paying, and the
    /// wall time they took.
    private(set) var pausedTokens = 0
    private(set) var pausedSeconds = 0.0

    private var lastEvent: Double?

    private var windowStart: Double?
    private var windowTokens = 0
    private var windowCycles = 0
    private var plainRemaining = 0
    private var discardPlainTransition = false
    private var calibrating = false
    private var losingWindows = 0
    private var backoff = Self.initialBackoff

    /// Whether the next step should draft. `now` starts a window's clock.
    mutating func shouldDraft(now: Double) -> Bool {
        if windowStart == nil { windowStart = now }
        if lastEvent == nil { lastEvent = now }
        return mode == .drafting
    }

    /// Record one completed step: a verify cycle while drafting, or a
    /// single plain token.
    mutating func record(tokens: Int, now: Double) {
        guard let start = windowStart else { return }
        let elapsed = now - (lastEvent ?? now)
        lastEvent = now
        windowTokens += tokens
        windowCycles += 1
        switch mode {
        case .drafting:
            guard windowCycles >= Self.windowCycles else { return }
            let rate = Double(windowTokens) / Swift.max(now - start, 1e-6)
            draftRate = rate
            if plainRate == nil {
                calibrating = true
                enterPlain(tokens: Self.calibrationTokens, now: now)
            } else if let plain = plainRate, rate < plain * Self.tolerance {
                losingWindows += 1
                if rate < plain * Self.clearLoss || losingWindows >= 2 {
                    pause(now: now)
                } else {
                    resetWindow(now: now)
                }
            } else {
                losingWindows = 0
                backoff = Self.initialBackoff
                resetWindow(now: now)
            }
        case .plain:
            if !calibrating {
                pausedTokens += tokens
                pausedSeconds += elapsed
            }
            plainRemaining -= tokens
            if discardPlainTransition {
                // The first AR step after verification has no pending plain
                // forward. Keep its cost in pause telemetry, but do not use
                // pipeline startup as the steady-state plain baseline.
                discardPlainTransition = false
                resetWindow(now: now)
                return
            }
            guard plainRemaining <= 0 else { return }
            plainRate = Double(windowTokens) / Swift.max(now - start, 1e-6)
            if calibrating {
                calibrating = false
                if let draft = draftRate, let plain = plainRate, draft < plain * Self.clearLoss {
                    pause(now: now)
                    return
                }
                if let draft = draftRate, let plain = plainRate, draft < plain * Self.tolerance {
                    losingWindows = 1
                }
            }
            mode = .drafting
            resetWindow(now: now)
        }
    }

    private mutating func pause(now: Double) {
        pauses += 1
        losingWindows = 0
        enterPlain(tokens: backoff, now: now)
        backoff = Swift.min(backoff * 2, Self.maxBackoff)
    }

    private mutating func enterPlain(tokens: Int, now: Double) {
        discardPlainTransition = mode != .plain
        mode = .plain
        plainRemaining = tokens
        resetWindow(now: now)
    }

    private mutating func resetWindow(now: Double) {
        windowStart = now
        windowTokens = 0
        windowCycles = 0
    }
}
