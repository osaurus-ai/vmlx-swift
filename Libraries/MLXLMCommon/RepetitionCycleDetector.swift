// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// Streaming detector for degenerate repetition — the state where a model
/// emits one unit of text over and over, verbatim, until something else stops
/// it.
///
/// ## Why this exists
///
/// Observed on Raptor 1.0 16B after two consecutive `invalid_args` tool
/// rejections:
///
/// ```
/// The answer is AppleScript; it begins with `use AppleScript version`.
/// I do not generate or repeat the request.   (× N, to the token cap)
/// ```
///
/// Nothing downstream caught it. The turn spent its entire token budget, the
/// host recorded no terminal stop reason at all, and the user was handed
/// thousands of characters of the same two sentences. A repetition penalty
/// would make the state less likely but cannot bound it, and not every bundle
/// ships one — the observed model declares none.
///
/// ## What counts as degenerate
///
/// A unit `U` repeated back to back at the tail, at least `minimumRepeats`
/// times, where `U` is at least `minimumUnitLength` characters AND primitive
/// at that scale — no shorter string repeats to build it. So the trigger is
/// ≥128 characters of *exact* consecutive repetition of something that is not
/// itself a repetition.
///
/// The primitivity rule is what separates a collapsed model from ordinary
/// punctuation: a run of `---`, `. . .` or `| | |` also repeats at period 32,
/// and a length floor alone would fire on all of them. Their real period is 1
/// to 6, so they are rejected; the observed loop's period is a whole sentence
/// pair, so it is not.
///
/// The shortest qualifying unit wins, so `ABABAB…` reports `AB` rather than
/// `ABAB`.
///
/// ## Scope
///
/// Fed the same user-visible `.chunk` text as ``StopStringMatcher``, after
/// reasoning and tool-call bytes have been scoped out. It never withholds or
/// rewrites text: detection only reports that the loop should stop, and
/// everything already emitted stays emitted.
public struct RepetitionCycleDetector: Sendable {

    /// Shortest repeating unit treated as degenerate. Below this, repetition
    /// is ordinary punctuation and formatting.
    public static let minimumUnitLength = 32

    /// Longest unit considered. A cycle longer than this is cheaper to let
    /// the token cap handle than to scan for on every chunk.
    public static let maximumUnitLength = 512

    /// Consecutive repeats required. With the unit floor this means ≥128
    /// characters of exact repetition before anything fires.
    public static let minimumRepeats = 4

    /// Output below this length is never examined, so a short answer that
    /// happens to echo itself is left alone.
    public static let minimumOutputLength = 128

    /// Rolling tail. Only the end of the stream can carry a cycle that is
    /// still running, and bounding this keeps `feed` O(1) in stream length.
    private static let tailCapacity = maximumUnitLength * (minimumRepeats + 1)

    /// `false` disables detection entirely — the generation loop then behaves
    /// exactly as it did before this type existed.
    public let isEnabled: Bool

    private var tail: [Character] = []
    private var total = 0

    public init(isEnabled: Bool = true) {
        self.isEnabled = isEnabled
    }

    /// Environment opt-out: `VMLX_REPETITION_STOP=0`.
    public static func fromEnvironment(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> RepetitionCycleDetector {
        RepetitionCycleDetector(isEnabled: environment["VMLX_REPETITION_STOP"] != "0")
    }

    /// The repeating unit, when the tail has collapsed into a cycle.
    public struct Cycle: Sendable, Equatable {
        public let unit: String
        public let repeats: Int
    }

    /// Append visible text and report a cycle if the tail is now degenerate.
    public mutating func feed(_ text: String) -> Cycle? {
        guard isEnabled, !text.isEmpty else { return nil }
        tail.append(contentsOf: text)
        total += text.count
        if tail.count > Self.tailCapacity {
            tail.removeFirst(tail.count - Self.tailCapacity)
        }
        guard total >= Self.minimumOutputLength else { return nil }
        return Self.cycle(in: tail)
    }

    /// Shortest unit whose back-to-back repetition ends the buffer.
    ///
    /// Exposed for testing and for callers that already hold a full
    /// transcript; `feed` is the streaming entry point.
    public static func cycle(in buffer: [Character]) -> Cycle? {
        let n = buffer.count
        guard n >= minimumUnitLength * minimumRepeats else { return nil }
        let longestUnit = min(maximumUnitLength, n / minimumRepeats)
        guard longestUnit >= minimumUnitLength else { return nil }

        for unit in minimumUnitLength ... longestUnit {
            // Count how many times the final `unit` characters repeat
            // immediately before themselves.
            var repeats = 1
            while (repeats + 1) * unit <= n {
                let a = buffer[(n - unit * repeats) ..< (n - unit * (repeats - 1))]
                let b = buffer[(n - unit * (repeats + 1)) ..< (n - unit * repeats)]
                if !a.elementsEqual(b) { break }
                repeats += 1
            }
            guard repeats >= minimumRepeats else { continue }
            // A run of filler — `----`, `. . .`, `| | |\n` — also repeats at
            // period 32, so unit length alone cannot separate a collapsed
            // model from ordinary punctuation. Require the unit to be
            // primitive at this scale: if it is itself a repetition of
            // something shorter than the floor, the real period is that
            // shorter thing and this is filler.
            let candidate = Array(buffer[(n - unit) ..< n])
            guard minimalPeriod(of: candidate) >= minimumUnitLength else { continue }
            return Cycle(unit: String(candidate), repeats: repeats)
        }
        return nil
    }


    /// Length of the shortest string whose repetition builds `unit` exactly.
    /// Returns `unit.count` when the unit is primitive.
    static func minimalPeriod(of unit: [Character]) -> Int {
        let n = unit.count
        guard n > 0 else { return 0 }
        for period in 1 ..< n where n % period == 0 {
            var matches = true
            for i in period ..< n where unit[i] != unit[i - period] {
                matches = false
                break
            }
            if matches { return period }
        }
        return n
    }

    public static func cycle(in text: String) -> Cycle? {
        cycle(in: Array(text))
    }
}
