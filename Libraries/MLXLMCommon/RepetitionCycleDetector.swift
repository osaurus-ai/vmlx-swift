// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// Text-pattern detector retained for clients that inspect repeated text.
///
/// A repeated pattern does not establish decoder failure: quotations, fixtures,
/// tables, and requested verbatim output can all repeat legitimately. This type
/// must not decide generation completion or rewrite the model's terminal reason.
/// The generation pipelines use EOS, configured stops, output limits, and caller
/// cancellation; they do not invoke this detector.
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

    /// Consecutive repeats of a LINE cycle required. Four back-to-back
    /// repeats of a block whose every line clears `minimumLineLength` is
    /// already unambiguous — at period 8 that is 32 identical substantial
    /// lines in a row.
    public static let minimumLineRepeats = 4

    /// Longest line CYCLE considered, in lines.
    ///
    /// Must comfortably exceed the number of lines it takes to pass the
    /// character scan's `maximumUnitLength` (512), or this scan cannot see its
    /// own target case: a unit longer than 512 characters necessarily spans
    /// several lines, and an earlier value of 4 here missed an 8-line block
    /// entirely. Cost is a handful of whole-line comparisons per completed
    /// line, not per chunk, so the headroom is cheap.
    public static let maximumLineCycle = 12

    /// A line must be at least this long, trimmed, before it can anchor a
    /// line cycle — and must contain a letter. Keeps table rules (`|---|`),
    /// separators, blank lines and `}` / `)` closers from ever qualifying.
    public static let minimumLineLength = 24

    /// Consecutive characters of SHORT-PERIOD filler that count as a collapse.
    ///
    /// The primitivity rule below excludes `....`, `----`, `| | |` on purpose:
    /// those are ordinary punctuation and formatting, and halting on them would
    /// be far worse than the bug. But that rule looks only at the UNIT and never
    /// at the VOLUME, so a model emitting `.` forever is indistinguishable from
    /// an ellipsis and runs to the token cap — reported live as "looping `.`
    /// output".
    ///
    /// Volume is what separates them. A horizontal rule is ~80 characters; a
    /// markdown table rule is ~80; the longest legitimate run of a <32-character
    /// unit in prose or code is far below this. 1024 consecutive characters of
    /// the same short unit is not formatting, it is a collapsed decoder.
    public static let fillerCollapseLength = 1024

    /// Rolling tail. Only the end of the stream can carry a cycle that is
    /// still running, and bounding this keeps `feed` O(1) in stream length.
    private static let tailCapacity = maximumUnitLength * (minimumRepeats + 1)

    /// `false` disables detection entirely — the generation loop then behaves
    /// exactly as it did before this type existed.
    public let isEnabled: Bool

    private var tail: [Character] = []
    private var total = 0
    /// Completed visible lines, most recent last. Bounded by the largest
    /// window the line scan can need.
    private var lines: [String] = []
    /// Bytes of the line currently being built (no newline seen yet).
    private var pendingLine: String = ""

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
        if let cycle = Self.cycle(in: tail) { return cycle }
        return feedLines(text)
    }

    /// Line-level cycle scan.
    ///
    /// The character scan bounds its unit at `maximumUnitLength` (512) because
    /// the cost of looking for longer units on every chunk is not worth it —
    /// the comment there says a longer cycle is "cheaper to let the token cap
    /// handle". That trade was costed against SCAN time, never against the
    /// user's wall clock, and the two diverge badly on slow hardware: a
    /// runaway that reaches a 16,384-token cap is ~2 minutes on a fast Mac and
    /// ~34 minutes at 8 tok/s on a base laptop. Reported live: a 36 KB file
    /// task where the model "kept writing the same line over and over" and the
    /// user gave up after 35 minutes.
    ///
    /// So this catches exactly the class the character scan gives up on — a
    /// repeating unit LONGER than 512 characters — and it does so at O(1) per
    /// completed line rather than O(unit x tail) per chunk, because it compares
    /// whole lines instead of every possible period.
    private mutating func feedLines(_ text: String) -> Cycle? {
        // Only completed lines can be compared; a line still being written
        // would false-negative on its own prefix.
        pendingLine += text
        guard pendingLine.contains("\n") else { return nil }
        var parts = pendingLine.components(separatedBy: "\n")
        pendingLine = parts.removeLast()
        lines.append(contentsOf: parts)
        let window = Self.maximumLineCycle * (Self.minimumLineRepeats + 1)
        if lines.count > window { lines.removeFirst(lines.count - window) }
        return Self.lineCycle(in: lines)
    }

    /// Shortest line cycle whose back-to-back repetition ends `lines`.
    ///
    /// Exposed for testing alongside `cycle(in:)`.
    public static func lineCycle(in lines: [String]) -> Cycle? {
        for period in 1 ... maximumLineCycle {
            guard lines.count >= period * minimumLineRepeats else { continue }
            let unit = Array(lines.suffix(period))
            // Every line in the unit must be substantial. Without this a run
            // of `|---|---|`, blank lines, or `    }` would qualify, and those
            // are ordinary formatting rather than a collapsed model.
            guard unit.allSatisfy({ line in
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                return trimmed.count >= minimumLineLength
                    && trimmed.contains(where: { $0.isLetter })
            }) else { continue }
            // A unit that is itself a repetition of a shorter block has that
            // shorter block as its real period — the same primitivity rule the
            // character scan applies, so `A A` is never reported as period 2.
            if period > 1, isRepetition(unit) { continue }
            var repeats = 1
            var offset = lines.count - period
            while offset >= period, Array(lines[(offset - period) ..< offset]) == unit {
                repeats += 1
                offset -= period
            }
            guard repeats >= minimumLineRepeats else { continue }
            return Cycle(unit: unit.joined(separator: "\n"), repeats: repeats)
        }
        return nil
    }

    /// True when `unit` is a whole number of copies of a shorter prefix.
    private static func isRepetition(_ unit: [String]) -> Bool {
        let n = unit.count
        for period in 1 ..< n where n % period == 0 {
            var matches = true
            for i in period ..< n where unit[i] != unit[i - period] {
                matches = false
                break
            }
            if matches { return true }
        }
        return false
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
            let period = minimalPeriod(of: candidate)
            guard period >= minimumUnitLength else {
                // Filler by unit, but check VOLUME before dismissing it: past
                // `fillerCollapseLength` characters of one short unit this is a
                // collapsed decoder, not punctuation.
                if let collapse = fillerCollapse(in: buffer, period: period) {
                    return collapse
                }
                continue
            }
            return Cycle(unit: String(candidate), repeats: repeats)
        }
        return nil
    }


    /// How far back the buffer is an unbroken repetition of its last `period`
    /// characters, reported as a cycle once it passes `fillerCollapseLength`.
    ///
    /// Deliberately measured in CHARACTERS rather than repeats: `.` repeated
    /// 1024 times and a 4-character unit repeated 256 times are the same
    /// pathology and the same wasted decode.
    static func fillerCollapse(in buffer: [Character], period: Int) -> Cycle? {
        guard period > 0, buffer.count >= fillerCollapseLength else { return nil }
        let unit = Array(buffer.suffix(period))
        var length = period
        var index = buffer.count - period
        while index >= period,
            Array(buffer[(index - period) ..< index]) == unit
        {
            length += period
            index -= period
        }
        guard length >= fillerCollapseLength else { return nil }
        return Cycle(unit: String(unit), repeats: length / period)
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
