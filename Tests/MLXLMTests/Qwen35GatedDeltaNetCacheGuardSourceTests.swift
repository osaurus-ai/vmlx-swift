// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

/// Coverage for the gated-delta-net restored-cache guard (#165).
///
/// A `MambaCache` slot restored with a stale shape — paged/hybrid restore or a
/// prefix commit — used to reach the `MLXArray` subscript rank precondition
/// inside `gatedDeltaUpdate` and abort the process. The guard validates both
/// slots and falls back to zero state instead.
///
/// This is a source-coverage suite rather than a behavioural one because
/// reaching the crash needs a mis-restored cache from the batch engine's solo
/// fast path, which cannot be constructed directly. What it pins is that the
/// guard exists in BOTH copies of the layer and keeps its shape contract:
/// Qwen35 is duplicated across MLXLLM (text) and MLXVLM (vision), and a fix
/// applied to one copy silently leaves the other crashing — the same
/// two-copies trap that hid the MLX test-lock wedge.
@Suite("Qwen35 GatedDeltaNet restored-cache guard")
struct Qwen35GatedDeltaNetCacheGuardSourceTests {

    private static let sources = [
        "Libraries/MLXLLM/Models/Qwen35.swift",
        "Libraries/MLXVLM/Models/Qwen35.swift",
    ]

    private func read(_ path: String) throws -> String {
        try String(contentsOfFile: path, encoding: .utf8)
    }

    @Test("both copies validate the conv slot before using it")
    func convSlotGuarded() throws {
        for path in Self.sources {
            let src = try read(path)
            #expect(
                src.contains("cacheState.ndim == 3"),
                "\(path) must reject a conv slot that is not rank 3")
            #expect(
                src.contains("cacheState.dim(2) == convDim"),
                "\(path) must reject a conv slot whose channel dim moved")
            #expect(
                src.contains(#"warnDiscardedCacheState(slot: "conv""#),
                "\(path) must report a discarded conv slot rather than resetting silently")
        }
    }

    @Test("both copies validate the recurrent slot's full [B, Hv, Dv, Dk] shape")
    func recurrentSlotGuarded() throws {
        for path in Self.sources {
            let src = try read(path)
            #expect(src.contains("cachedState.ndim == 4"), "\(path)")
            #expect(src.contains("cachedState.dim(1) == numVHeads"), "\(path)")
            #expect(src.contains("cachedState.dim(2) == headVDim"), "\(path)")
            #expect(src.contains("cachedState.dim(3) == headKDim"), "\(path)")
            #expect(
                src.contains(#"warnDiscardedCacheState(slot: "recurrent""#),
                "\(path) must report a discarded recurrent slot")
        }
    }

    /// The warning must be reachable ONLY when a slot was actually present and
    /// malformed. Warning on an absent cache would fire on every cold start and
    /// train readers to ignore it.
    @Test("a missing cache is not reported as a discarded slot")
    func absentCacheIsNotAWarning() throws {
        for path in Self.sources {
            let src = try read(path)
            #expect(
                src.contains("if let cacheState = cache?[0] {"),
                "\(path) must bind the slot before warning, so an absent cache stays silent")
        }
    }

    /// One-shot per process: a per-token warning inside the decode loop would
    /// flood stderr for the whole generation.
    @Test("the warning is emitted once per process")
    func warningIsOneShot() throws {
        for path in Self.sources {
            let src = try read(path)
            #expect(src.contains("didWarnDiscardedCache"), "\(path)")
            #expect(
                src.contains("guard !didWarnDiscardedCache else { return }"),
                "\(path) must return after the first warning")
        }
    }
}
