// Copy drafts for the native-MTP verify cycle — Swift port of vMLX Python `native_mtp_copy.py`
// (SuffixCopyProposer; the reference SuffixLookupProposer). On repetitive, quoted, edited or structured text
// the next tokens very often repeat a span that already occurs earlier in the prompt or reply; proposing that
// span as the verify window lands far more often than the MTP head's chained drafts, whose per-depth
// acceptance compounds.
//
// HOW: the proposer keeps the request's confirmed token stream (prompt + every confirmed token) and an
// incremental index from each 4-gram to its most recent end positions (last 4 kept). A proposal needs the
// current suffix to match an earlier occurrence for at least `minMatch` tokens (default 8 — short n-gram
// matches are mostly coincidence and a wrong copy displaces the head's own drafts). The continuation after the
// longest such match is proposed, up to `width` tokens. Width starts at 3 (4 verify rows), jumps to `maxWidth`
// (default 7 → 8 rows, the row-exact verify limit) after a fully accepted window and falls back to 3 after a
// window that lands under half. Two consecutive windows whose first copied token missed silence it for 16
// proposals.
//
// EXACTNESS: copies change only WHICH tokens fill the verify window. The verify forward, greedy acceptance
// (exact match against the target's own argmax per row), bonus/correction and staged rollback are the
// existing native-MTP ones, and verify windows up to 8 rows are row-exact, so greedy output is unchanged.
// Greedy only (the sampled exact-pq path needs draft probabilities a copy does not have).
// Env: VMLX_NATIVE_MTP_COPY=0 disables; VMLX_NATIVE_MTP_COPY_MIN_MATCH (>=4), VMLX_NATIVE_MTP_COPY_MAX (1...7).

import Foundation

struct NativeMTPCopyProposer {
    static let ngram = 4
    static let startWidth = 3
    static let positionsKept = 4
    static let maxExtend = 256
    static let silenceCycles = 16

    static var enabled: Bool {
        let raw = (ProcessInfo.processInfo.environment["VMLX_NATIVE_MTP_COPY"] ?? "1").lowercased()
        return !["0", "false", "off", "no"].contains(raw)
    }

    private static func envInt(_ name: String, _ fallback: Int, minimum: Int, maximum: Int) -> Int {
        guard let raw = ProcessInfo.processInfo.environment[name], let v = Int(raw) else {
            return fallback
        }
        return Swift.min(maximum, Swift.max(minimum, v))
    }

    struct Stats {
        var proposals = 0
        var cycles = 0
        var drafted = 0
        var accepted = 0
        var firstMisses = 0
        var silenced = 0
    }

    let minMatch: Int
    let maxWidth: Int
    private(set) var tokens: [Int] = []
    private var index: [[Int]: [Int]] = [:]
    private(set) var width = NativeMTPCopyProposer.startWidth
    private var missStreak = 0
    private var silentFor = 0
    private(set) var stats = Stats()

    init(prompt: [Int]) {
        minMatch = Self.envInt(
            "VMLX_NATIVE_MTP_COPY_MIN_MATCH", 8, minimum: Self.ngram, maximum: 4096)
        maxWidth = Self.envInt("VMLX_NATIVE_MTP_COPY_MAX", 7, minimum: 1, maximum: 7)
        tokens.reserveCapacity(prompt.count + 1024)
        append(prompt)
    }

    mutating func append<S: Sequence>(_ new: S) where S.Element == Int {
        for t in new {
            tokens.append(t)
            let i = tokens.count - 1
            guard i >= Self.ngram - 1 else { continue }
            let key = Array(tokens[(i - Self.ngram + 1) ... i])
            if var slot = index[key] {
                slot.append(i)
                if slot.count > Self.positionsKept { slot.removeFirst() }
                index[key] = slot
            } else {
                index[key] = [i]
            }
        }
    }

    /// Up to min(width, room) copied token ids, or [] when no earlier span matches the suffix well enough.
    mutating func propose(room: Int) -> [Int] {
        let w = Swift.min(width, maxWidth, room)
        let n = tokens.count
        guard w > 0, n >= Swift.max(Self.ngram, minMatch) + 1 else { return [] }
        if silentFor > 0 {
            silentFor -= 1
            return []
        }
        guard let candidates = index[Array(tokens[(n - Self.ngram)...])] else { return [] }
        var bestPos = -1
        var bestLen = 0
        for p in candidates.reversed() where p < n - 1 {
            var length = Self.ngram
            let limit = Swift.min(Self.maxExtend, p + 1)
            while length < limit, tokens[p - length] == tokens[n - 1 - length] { length += 1 }
            if length > bestLen {
                bestPos = p
                bestLen = length
            }
        }
        guard bestPos >= 0, bestLen >= minMatch else { return [] }
        let start = bestPos + 1
        let end = Swift.min(n, start + w)
        guard start < end else { return [] }
        stats.proposals += 1
        return Array(tokens[start ..< end])
    }

    /// A verified copy window: `drafted` copied tokens, `accepted` of them confirmed.
    mutating func observe(drafted: Int, accepted: Int) {
        stats.cycles += 1
        stats.drafted += drafted
        stats.accepted += accepted
        if drafted > 0, accepted == drafted {
            width = maxWidth
        } else if drafted > 0, 2 * accepted < drafted {
            width = Self.startWidth
        }
        if accepted == 0 {
            stats.firstMisses += 1
            missStreak += 1
            if missStreak >= 2 {
                silentFor = Self.silenceCycles
                missStreak = 0
                stats.silenced += 1
            }
        } else {
            missStreak = 0
        }
    }
}
