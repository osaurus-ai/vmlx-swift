// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// The GLM-4V vision path reshapes each frame to (h/merge, merge, w/merge, merge). A frame the
// merge size does not divide makes the element count disagree and MLX traps; a merge size of zero
// divides by zero. Neither is recoverable, but a trap ends the process with a message about tensor
// shapes rather than about the bundle that caused it.
//
// The guard runs at `prepare`, which already throws — so the non-throwing forward path keeps its
// signatures and simply relies on an invariant established before it runs.

import Foundation
import MLXLMCommon
import Testing

@testable import MLXVLM

@Suite("GLM-4V grid invariants")
struct Glm4GridInvariantTests {

    @Test("a divisible grid passes")
    func divisiblePasses() throws {
        try Glm4SharedVision.validateGrid(
            [THW(1, 32, 48), THW(1, 16, 16)], spatialMergeSize: 2)
    }

    @Test("an empty frame list passes — text-only requests are not vision requests")
    func emptyPasses() throws {
        try Glm4SharedVision.validateGrid([], spatialMergeSize: 2)
    }

    /// The message must name the bundle-level cause. "shape mismatch" from inside a reshape sends
    /// the reader to the wrong layer entirely.
    @Test("a non-divisible frame throws, naming the frame and the merge size")
    func nonDivisibleThrows() {
        #expect(throws: (any Error).self) {
            try Glm4SharedVision.validateGrid([THW(1, 33, 48)], spatialMergeSize: 2)
        }
        do {
            try Glm4SharedVision.validateGrid([THW(1, 32, 48), THW(1, 15, 20)], spatialMergeSize: 2)
            Issue.record("expected a throw")
        } catch {
            let text = "\(error)"
            #expect(text.contains("15"), "should name the offending extent: \(text)")
            #expect(text.contains("frame 1"), "should name WHICH frame: \(text)")
            #expect(text.contains("2"), "should name the merge size: \(text)")
        }
    }

    @Test("a zero merge size throws instead of dividing by zero")
    func zeroMergeThrows() {
        #expect(throws: (any Error).self) {
            try Glm4SharedVision.validateGrid([THW(1, 32, 32)], spatialMergeSize: 0)
        }
    }

    /// Both towers must establish the invariant, not just the one that was debugged.
    @Test("both GLM towers validate before touching the vision path")
    func bothTowersGuard() throws {
        for file in ["Libraries/MLXVLM/Models/Glm4v.swift", "Libraries/MLXVLM/Models/Glm4vMoe.swift"] {
            let source = try #require(
                try? String(contentsOfFile: file, encoding: .utf8), "missing \(file)")
            #expect(
                source.contains("Glm4SharedVision.validateGrid("),
                "\(file) does not establish the grid invariant")
        }
    }
}
