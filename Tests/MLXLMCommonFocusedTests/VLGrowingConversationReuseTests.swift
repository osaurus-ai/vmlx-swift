// Copyright © 2026 Osaurus AI. All rights reserved.
//
// What a GROWING VL conversation can actually reuse from the prefix cache.
//
// The primitives are already covered elsewhere: `MediaCachePlaceholderTests`
// pins `cacheHitSuffixContainsMediaPlaceholder`, and
// `CacheCoordinatorMediaSaltTests` pins `computeMediaSalt`. Neither states
// the property a user feels, which is per-turn: *which turns of a real
// multi-image chat re-prefill the whole conversation, and which resume.*
//
// Two independent mechanisms decide that, and they must be read together:
//
//   1. `computeMediaSalt` fingerprints the request's WHOLE concatenated pixel
//      tensor (`ProcessedImage.pixels` is documented as "Concatenated pixels
//      from one or more images"). Appending an image changes the salt for the
//      entire prompt, so no boundary stored by an earlier turn can match --
//      `CacheCoordinator` passes the salt to `DiskCache.fetch` on every
//      candidate boundary probe.
//   2. Even given a matching salt, a hit whose REMAINING suffix still holds
//      media placeholders is rolled back to a full prefill, in all four
//      decode paths (`Evaluate`, `BatchEngine`, `DFlash2TokenIterator`,
//      `NativeMTPTokenIterator`). Media embeddings are substituted onto
//      placeholder positions by order across the whole prompt, so prefilling
//      only a suffix would map the wrong image onto them.
//
// Net effect, and the thing to know before optimizing: a turn that INTRODUCES
// new media pays a full re-prefill of the conversation (vision tower
// included). Turns that only add text after it resume normally. A chat that
// varies modality every turn therefore never resumes, which is the shape that
// degrades over long context.
//
// These are assertions about today's behaviour. If someone makes
// media-introducing turns resumable -- prefix-scoped salting plus a
// media-consumed offset threaded into `prepare` -- these expectations SHOULD
// change, deliberately, and the rollback log line
// "rolling back to full prefill (media placeholder tokens remain in
// cache-hit suffix)" is how you confirm it live.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("VL growing-conversation prefix reuse")
struct VLGrowingConversationReuseTests {

    /// Placeholder id standing in for `<image>`.
    private static let imageToken = 27

    /// `count` distinct 1x3x2x2 images concatenated on the leading axis,
    /// matching how a processor hands multiple images to the model.
    private static func pixels(count: Int) -> MLXArray {
        let perImage = 3 * 2 * 2
        var values: [Float] = []
        for image in 0..<count {
            values.append(contentsOf: (0..<perImage).map { Float(image * 100 + $0) })
        }
        let array = MLXArray(values).reshaped([count, 3, 2, 2])
        MLX.eval(array)
        return array
    }

    private static func input(tokens: [Int32], imageCount: Int) -> LMInput {
        LMInput(
            text: .init(tokens: MLXArray(tokens)),
            image: .init(pixels: pixels(count: imageCount)),
            mediaTokenIds: [imageToken])
    }

    /// Turn 2 is a text follow-up about the same picture: the image tensor is
    /// unchanged, so the salt matches, and the boundary at the end of turn 1
    /// leaves a text-only suffix. This turn resumes -- the common VL case.
    @Test("a text follow-up after an image turn resumes from the cached prefix")
    func textFollowUpAfterImageTurnResumes() {
        FocusedMLXTestSupport.withLock {
            let turn1 = Self.input(tokens: [1, 27, 27, 2], imageCount: 1)
            let turn2 = Self.input(tokens: [1, 27, 27, 2, 5, 6], imageCount: 1)

            #expect(computeMediaSalt(for: turn1) == computeMediaSalt(for: turn2))

            // Boundary = end of turn 1 (4 tokens); suffix is [5, 6].
            #expect(!turn2.cacheHitSuffixContainsMediaPlaceholder([5, 6]))
        }
    }

    /// The turn that adds a SECOND image cannot resume, for two separate
    /// reasons. Assert both, so removing one and declaring victory is not
    /// possible: the salt alone still leaves the suffix rollback, and lifting
    /// the suffix rollback alone still leaves an unmatchable salt.
    @Test("adding a second image blocks reuse twice over")
    func addingASecondImageBlocksReuseTwice() {
        FocusedMLXTestSupport.withLock {
            let turn1 = Self.input(tokens: [1, 27, 27, 2], imageCount: 1)
            let turn2 = Self.input(tokens: [1, 27, 27, 2, 5, 27, 27, 6], imageCount: 2)

            // (1) Whole-tensor salt: turn 1's stored boundaries are keyed
            // under a salt turn 2 never presents, so every probe misses.
            #expect(computeMediaSalt(for: turn1) != computeMediaSalt(for: turn2))

            // (2) Suffix rollback: even at the turn-1 boundary the remaining
            // suffix carries the new image's placeholders.
            #expect(turn2.cacheHitSuffixContainsMediaPlaceholder([5, 27, 27, 6]))
        }
    }

    /// Reuse is not lost permanently -- it resumes on the next text-only turn,
    /// because that turn presents the same two-image tensor as the turn before
    /// it. The cost is per media-introducing turn, not cumulative.
    @Test("reuse resumes on the text turn after a media-introducing turn")
    func reuseResumesAfterTheMediaIntroducingTurn() {
        FocusedMLXTestSupport.withLock {
            let turn2 = Self.input(tokens: [1, 27, 27, 2, 5, 27, 27, 6], imageCount: 2)
            let turn3 = Self.input(tokens: [1, 27, 27, 2, 5, 27, 27, 6, 7, 8], imageCount: 2)

            #expect(computeMediaSalt(for: turn2) == computeMediaSalt(for: turn3))
            #expect(!turn3.cacheHitSuffixContainsMediaPlaceholder([7, 8]))
        }
    }

    /// A boundary that lands INSIDE a placeholder run must never be treated as
    /// capturable. This is the media analogue of the alignment lesson: the
    /// interesting boundary is the one that does not divide evenly.
    @Test("a boundary inside the placeholder run is not capturable")
    func boundaryInsideThePlaceholderRunIsNotCapturable() {
        FocusedMLXTestSupport.withLock {
            let prompt: [Int] = [1, 27, 27, 27, 2]
            let turn = Self.input(tokens: prompt.map(Int32.init), imageCount: 1)

            // Splitting the run at 2 or 3 leaves placeholders on both sides.
            #expect(!turn.canCaptureHybridStripBoundary(promptTokenIds: prompt, boundary: 2))
            #expect(!turn.canCaptureHybridStripBoundary(promptTokenIds: prompt, boundary: 3))

            // After the complete run the boundary has a stable token identity.
            #expect(turn.canCaptureHybridStripBoundary(promptTokenIds: prompt, boundary: 4))

            // Before the run there is no media in the prefix to anchor it.
            #expect(!turn.canCaptureHybridStripBoundary(promptTokenIds: prompt, boundary: 1))
        }
    }

    /// Audio and video ride the same two mechanisms as images, so a chat that
    /// alternates modality introduces new media on every turn and therefore
    /// never resumes. Pin that explicitly -- it is the "variating
    /// multimodality per turn" shape.
    @Test("alternating modality changes the salt every turn")
    func alternatingModalityChangesTheSaltEveryTurn() {
        FocusedMLXTestSupport.withLock {
            let waveform = MLXArray((0..<8).map { Float($0) }).reshaped([1, 8])
            MLX.eval(waveform)

            let imageTurn = Self.input(tokens: [1, 27, 2], imageCount: 1)
            let plusAudioTurn = LMInput(
                text: .init(tokens: MLXArray([Int32(1), 27, 2, 5, 27])),
                image: .init(pixels: Self.pixels(count: 1)),
                audio: .init(waveform: waveform),
                mediaTokenIds: [Self.imageToken])

            #expect(computeMediaSalt(for: imageTurn) != computeMediaSalt(for: plusAudioTurn))
            #expect(plusAudioTurn.cacheHitSuffixContainsMediaPlaceholder([5, 27]))
        }
    }
}
