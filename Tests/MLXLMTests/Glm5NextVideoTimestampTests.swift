// Copyright © 2026 osaurus-eval contributors
// SPDX-License-Identifier: MIT
//
// GLM-5.3's half of the video timestamp contract, and the assertion that the two families must
// never converge.
//
// GLM-4V and GLM-5.3 render video identically — both rewrite a video placeholder into a run of
// image blocks, one per temporal patch, each followed by that frame's timestamp, and their
// `replace_video_token` in transformers is byte-identical. Their `replace_frame_token_id` is not:
//
//     GLM-4V    <|end_of_image|>2
//     GLM-5.3   <|end_of_image|>2.5 seconds
//
// This is the kind of difference that survives everything else. The prompt is well-formed under
// either format, every shape lines up, the tower runs, and the model answers fluently — from a
// prompt that is not the one its weights were trained against. Nothing downstream can detect it.
//
// The GLM-4V half is pinned next to the GLM-4V implementation, in `Glm4vVideoTimestampTests`. This
// file carries the GLM-5.3 half plus the cross-family inequality, because the inequality can only
// be stated once BOTH processors exist and this is the later of the two.

import Foundation
import MLXVLM
import Testing

@Suite("GLM-5.3 video frame timestamps")
struct Glm5NextVideoTimestampTests {

    @Test("GLM-5.3 writes one decimal place and the word seconds")
    func glm5Format() {
        #expect(Glm5NextProcessor.frameTimestampMarkup(0) == "0.0 seconds")
        #expect(Glm5NextProcessor.frameTimestampMarkup(2) == "2.0 seconds")
        #expect(Glm5NextProcessor.frameTimestampMarkup(2.5) == "2.5 seconds")
        // One decimal, so a finer timestamp is rounded rather than carried.
        #expect(Glm5NextProcessor.frameTimestampMarkup(2.46) == "2.5 seconds")
    }

    /// The point of the pair: the two must NOT agree.
    ///
    /// Written as an explicit inequality because the tempting refactor — one shared helper, since
    /// everything else about the two paths is identical — is exactly the bug. If a later change
    /// makes them the same, this fails and says why.
    @Test("the two families must disagree, at every timestamp")
    func theyMustDiffer() {
        for seconds in stride(from: 0.0, through: 10.0, by: 0.5) {
            #expect(
                GlmOcrProcessor.frameTimestampMarkup(seconds)
                    != Glm5NextProcessor.frameTimestampMarkup(seconds),
                "the families produced the same markup at \(seconds)s; one of them is now wrong")
        }
    }
}
