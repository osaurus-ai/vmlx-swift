// Copyright © 2026 osaurus-eval contributors
// SPDX-License-Identifier: MIT
//
// GLM-4V rewrites a video placeholder into a RUN OF IMAGE BLOCKS, one per temporal patch, each
// followed by that frame's timestamp. The timestamp format is the one thing about that rewrite which
// is not self-evident from the surrounding code, and it is invisible to everything downstream: the
// prompt is well-formed under any plausible format, every shape lines up, the tower runs, and the
// model answers fluently — from a prompt that is not the one its weights were trained against.
//
// So it is pinned here, at the only place it exists.
//
// GLM-5.3 renders video the same way except for this one token run, and pinning THAT — along with
// the assertion that the two families must never converge — belongs with the GLM-5.3
// implementation, which is where `Glm5NextProcessor` is introduced. Keeping the two halves with
// their own implementations is what lets this file stand on its own.

import Foundation
import MLXVLM
import Testing

@Suite("GLM-4V video frame timestamps")
struct Glm4vVideoTimestampTests {

    @Test("GLM-4V writes a bare integer second")
    func glm4vFormat() {
        #expect(GlmOcrProcessor.frameTimestampMarkup(0) == "0")
        #expect(GlmOcrProcessor.frameTimestampMarkup(2) == "2")
        // Truncated, not rounded: the reference calls int() on the float.
        #expect(GlmOcrProcessor.frameTimestampMarkup(2.5) == "2")
        #expect(GlmOcrProcessor.frameTimestampMarkup(2.9) == "2")
    }
}
