// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Pins the aliasing semantics the block-diffusion no-empty-response guard
// depends on.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// The first-canvas EOS guard in `BlockDiffusionTokenIterator` suppresses EOS by writing
/// **in place**:
///
/// ```swift
/// var sampled = processed
/// sampled[0, 0, eos] = MLXArray(Float(-1e9))
/// ```
///
/// That is load-bearing, not incidental. `MLXArray` is a `final class`, so `sampled` is a second
/// reference to the same buffer and the write also reaches `processed` — which is what later
/// becomes `selfConditioning`. Without that reach, the model re-asserts its EOS preference on the
/// next denoising step and the canvas collapses to empty again: measured at boolq gen-nothink
/// **70.0% (35/50)** in place versus **0.0% with 8/8 extraction failures** when the same mask is
/// built out of place.
///
/// So this file guards a dependency that is invisible at the call site. If `MLXArray` ever gains
/// value semantics for subscript assignment — or someone "cleans up" the guard into
/// `sampled = processed + bias` — the fix silently degrades to producing no output at all, with no
/// other test failing. The whole point is that the *aliasing* is the mechanism.
///
/// Note the inverse trap also exists and is equally real: when the goal is to keep the caller's
/// logits untouched, a mask MUST be built out of place (`MLX.where` over an `arange`), because this
/// same aliasing makes a "copy" not a copy. Both rules follow from `MLXArray` being a class.
@Suite("Block-diffusion EOS guard relies on MLXArray subscript aliasing")
struct BlockDiffusionEOSGuardAliasingTests {

    @Test("subscript assignment through a second binding mutates the original")
    func subscriptAssignmentAliasesTheOriginal() {
        let mlxTestLock = lockSerializedMLXTest()
        defer { mlxTestLock.unlock() }

        let processed = MLXArray(converting: [1.0, 2.0, 3.0, 4.0]).reshaped(1, 1, 4)
        let eos = 2

        var sampled = processed
        sampled[0, 0, eos] = MLXArray(Float(-1e9))

        // The guard's correctness rests on this: `processed` — the value that goes on to become
        // self-conditioning — must carry the suppression even though it was never assigned to.
        let carried = processed[0, 0, eos].item(Float.self)
        #expect(
            carried == Float(-1e9),
            """
            MLXArray subscript assignment no longer aliases: the block-diffusion first-canvas EOS \
            guard now leaves self-conditioning unmasked, which measured 0% output (8/8 extraction \
            failures) on boolq gen-nothink. The guard must be rewritten to mask self-conditioning \
            explicitly.
            """)

        // Untouched positions must be unchanged — the guard suppresses exactly one id.
        #expect(processed[0, 0, 0].item(Float.self) == 1.0)
        #expect(processed[0, 0, 3].item(Float.self) == 4.0)
    }

    /// The complement, so the two rules are pinned together rather than one being "discovered"
    /// again later: an out-of-place mask leaves the source intact. This is the form to use when a
    /// caller still owns the logits.
    @Test("out-of-place masking leaves the source untouched")
    func outOfPlaceMaskingDoesNotAliasTheSource() {
        let mlxTestLock = lockSerializedMLXTest()
        defer { mlxTestLock.unlock() }

        let logits = MLXArray(converting: [1.0, 2.0, 3.0, 4.0]).reshaped(1, 1, 4)
        let ids = MLXArray(Int32(0) ..< Int32(4)).reshaped(1, 1, 4)
        let masked = MLX.where(ids .== MLXArray(Int32(2)), MLXArray(Float(-1e9)), logits)

        #expect(masked[0, 0, 2].item(Float.self) == Float(-1e9))
        #expect(logits[0, 0, 2].item(Float.self) == 3.0)
    }
}
