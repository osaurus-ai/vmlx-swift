// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

/// Pins the hybrid clamp on the growing-chat cache gate.
///
/// A hybrid topology with a canonical strip boundary deliberately suppresses every other cache
/// boundary (`usesCanonicalHybridBoundary` in `BatchEngine` / `TokenIterator`): persisting the
/// exact prompt and post-answer snapshots as well measured several near-identical full
/// serializations per turn — hundreds of MB each on Bonsai — to buy reuse the next turn does not
/// need.
///
/// The gate therefore must not ask a hybrid for a boundary the shipped policy never stores. Before
/// the clamp it did, and the verdict came down to a chat-template detail rather than cache health:
///
/// | model | turn-2 common prefix | branch taken | verdict |
/// |---|---|---|---|
/// | `Bonsai-27b-Ternary` | 21/23 — diverges | relaxed (stored max = 18) | passed |
/// | `Ornith-1.0-9B-MXFP8` | 25/25 — no divergence | strict (full prompt = 25) | **failed** at `matched=18` |
/// | `Nemotron-Audex-30B` | no divergence | strict | **failed** at `matched=41, expected 47` |
///
/// Same policy, opposite verdicts. The clamp is `min(unclamped, stored max)` and applies ONLY when
/// `coordinator.isHybrid`, so dense/rotating models keep the full-prompt expectation — and a hybrid
/// that regresses below its own stored boundary still fails, which is where the gate has teeth.
@Suite("growing-chat cache gate clamps hybrid expectations")
struct GrowingChatCacheHybridGateSourceTests {

    private func benchSource() throws -> String {
        try String(contentsOfFile: "RunBench/Bench.swift", encoding: .utf8)
    }

    @Test("the hybrid ceiling is derived from the stored boundaries and gated on isHybrid")
    func hybridCeilingIsGatedOnIsHybrid() throws {
        let bench = try benchSource()
        #expect(
            bench.contains(
                "let canonicalHybridCeiling = coordinator.isHybrid ? cachePrefixTokenCounts.max() : nil"
            ),
            """
            The hybrid ceiling must come from the STORED boundaries and be gated on `isHybrid`. \
            Deriving it from the prompt length would re-ask hybrids for a boundary the canonical \
            strip policy never persists; dropping the `isHybrid` gate would relax dense models too.
            """)
    }

    @Test("the expectation is clamped, not replaced")
    func expectationIsClampedRatherThanReplaced() throws {
        let bench = try benchSource()
        // `min` of the two, so the clamp can only ever LOWER the bar to what is storable — it can
        // never raise it, and it cannot mask a hybrid falling below its own stored boundary.
        #expect(bench.contains("Swift.min(unclampedPromptBoundary, $0)"))
        #expect(bench.contains("} ?? unclampedPromptBoundary"))
        // The unclamped computation must survive: dense/rotating models keep the strict path.
        #expect(bench.contains("let unclampedPromptBoundary = promptCommon < promptTokens.count"))
    }
}
