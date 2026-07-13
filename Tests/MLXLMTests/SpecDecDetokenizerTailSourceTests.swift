// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@testable import MLXLMCommon

/// `NaiveStreamingDetokenizer.next()` emits nothing until the decoded segment
/// exceeds `trailingHoldbackCharacters` (24) and then always trims that many
/// characters off the end — it is holding them back for a stable boundary. Only
/// `flush()` releases them.
///
/// `SpecDecStream.flush` accepted `detokenizer: inout` and never used it. So on
/// that path an answer shorter than the holdback rendered as *nothing at all*,
/// and a longer one silently lost its final ≤24 characters — including any
/// reasoning or tool-call close marker stranded in the tail.
///
/// The path is currently unreachable in production (`resolvedMTPDraftStrategy`
/// only ever returns `.nativeMTP`, and `usesBlockDiffusion` is false for it), so
/// this was a latent trap rather than a live bug. These guards keep it that way.
@Suite("SpecDec flushes the detokenizer's held tail")
struct SpecDecDetokenizerTailSourceTests {

    private static func source(_ relative: String) throws -> String {
        let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(relative)
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// The holdback is what makes the missing flush destructive. If someone
    /// lowers it to 0 this suite's premise changes, so pin it.
    @Test("the detokenizer withholds a trailing tail that only flush() releases")
    func detokenizerWithholdsATail() {
        #expect(NaiveStreamingDetokenizer.trailingHoldbackCharacters > 0)
    }

    /// The actual regression: `flush` must consume the detokenizer it is handed.
    @Test("SpecDecStream.flush drains the detokenizer")
    func flushDrainsTheDetokenizer() throws {
        let src = try Self.source("Libraries/MLXLMCommon/SpecDec/SpecDecStream.swift")

        guard let fn = src.range(of: "private static func flush(") else {
            Issue.record("`SpecDecStream.flush` is gone — rewrite this guard")
            return
        }
        // The function runs until the next `private static func` / end of type.
        let rest = src[fn.upperBound...]
        let end = rest.range(of: "\n    private static func")?.lowerBound ?? rest.endIndex
        let body = String(rest[..<end])

        #expect(
            body.contains("detokenizer.flush()"),
            """
            `SpecDecStream.flush` takes `detokenizer: inout` and must actually drain it. \
            Without this the last \(NaiveStreamingDetokenizer.trailingHoldbackCharacters) \
            characters of every answer are lost, and a short answer renders as nothing. \
            Body was:
            \(body)
            """)
    }

    /// The tail must go through the parsers, not straight to the stream: a
    /// reasoning or tool-call close marker stranded in it only lands if it does.
    @Test("the drained tail is routed through the reasoning/tool parsers")
    func drainedTailGoesThroughTheParsers() throws {
        let src = try Self.source("Libraries/MLXLMCommon/SpecDec/SpecDecStream.swift")

        guard let fn = src.range(of: "private static func flush(") else {
            Issue.record("`SpecDecStream.flush` is gone — rewrite this guard")
            return
        }
        let rest = src[fn.upperBound...]
        let end = rest.range(of: "\n    private static func")?.lowerBound ?? rest.endIndex
        let body = String(rest[..<end])

        guard let flushAt = body.range(of: "detokenizer.flush()"),
            let routeAt = body.range(of: "routeChunk(")
        else {
            Issue.record("expected the drained tail to be handed to `routeChunk`")
            return
        }
        #expect(
            flushAt.lowerBound < routeAt.lowerBound,
            "drain the detokenizer, THEN route its tail through the parsers")
    }
}
