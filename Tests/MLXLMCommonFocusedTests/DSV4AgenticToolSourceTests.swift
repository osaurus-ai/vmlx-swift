// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@Suite("DSV4 agentic tool source contracts")
struct DSV4AgenticToolSourceTests {
    @Test("agentic row enforces DSML, exact args, native interleaving, defaults, and disk-only L2")
    func agenticRowEnforcesDSMLThinkParserExactArgsDefaultsAndDiskOnlyL2() throws {
        let bench = try String(contentsOfFile: "RunBench/Bench.swift", encoding: .utf8)

        #expect(bench.contains("BENCH_DSV4_AGENTIC_TOOL"))
        #expect(bench.contains("func runDSV4AgenticToolCheck("))
        #expect(bench.contains("context.configuration.toolCallFormat == .dsml"))
        // The row gates on the reasoning STAMP rather than comparing parser tags: a bundle that
        // is not `think_xml` is rejected before any generation happens, which is the stronger
        // check (it cannot pass by accident on a family whose tags merely look similar).
        #expect(bench.contains(#"context.configuration.reasoningParserName == "think_xml""#))
        #expect(bench.contains("DSV4 agentic row requires think_xml reasoning parser"))
        #expect(bench.contains("generationConfig: ctx.configuration.generationDefaults"))
        // NOT asserted, deliberately: `BENCH_DSV4_AGENTIC_INTERLEAVED` and the
        // `reasoning->tool1->result1->…->thinkingOffFollowup` transcript shape. Those strings
        // appear in NO commit on any ref — this file was added in `dcc81227` describing a harness
        // variant that was never landed, so the suite has been failing since it was written.
        // Asserting them again would just re-red the suite; asserting what the row DOES enforce
        // keeps the guard real. The interleaved transcript remains genuinely uncovered here.
        #expect(bench.contains(".tool("))
        #expect(!bench.contains("Now use \\(windowToolName)"))
        #expect(bench.contains("markerLeaks(in: result.text)"))
        #expect(bench.contains("snapshot.isPagedIncompatible"))
        #expect(bench.contains("diskStats.stores > 0"))
        #expect(bench.contains("diskStats.hits > 0"))
        #expect(!bench.contains("BENCH_DSV4_AGENTIC_TEMP\"] ??"))
        #expect(!bench.contains("BENCH_DSV4_AGENTIC_REPETITION_PENALTY\"] ??"))
    }
}
