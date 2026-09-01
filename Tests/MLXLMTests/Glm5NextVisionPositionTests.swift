// Copyright © 2026 osaurus-eval contributors
// SPDX-License-Identifier: MIT
//
// The vision tower has to know WHERE each patch is.
//
// It did not. `Glm5NextVisionRotaryEmbedding` was missing entirely, and no existing test could have
// noticed: a rotary embedding has no learned parameters, so the checkpoint-versus-parameter-tree
// comparison — which this file has, in BOTH directions — is structurally blind to its absence. The
// tower loaded every shipped tensor, ran, produced finite non-constant features of the right shape,
// and passed an end-to-end test that fed it a real image. It simply had no idea what was next to
// what, and the model described the wrong picture.
//
// The property that catches it is permutation sensitivity. Self-attention without positional
// information is permutation-EQUIVARIANT: shuffle the input rows and the output rows shuffle with
// them, unchanged. Add positions and that stops being true. So the test is not "does it produce
// numbers" but "does shuffling the patches change what each patch becomes".

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing

@Suite("GLM-5.3 vision positions")
struct Glm5NextVisionPositionTests {

    /// Patches arrive BLOCK-MAJOR: `QwenVL.patchify` puts each `merge x merge` neighbourhood in
    /// consecutive positions, so the coordinates must be laid out the same way. Numbering row by
    /// row would give every patch a plausible, wrong coordinate.
    @Test("position ids are laid out block-major, not raster")
    func positionIdsAreBlockMajor() throws {
        try MLXMetalTestLock.withLock {
            // A 4x4 patch grid with merge 2: four 2x2 blocks, each contributing four positions.
            let ids = Glm5NextVisionRotary.positionIds(grid: THW(1, 4, 4), mergeSize: 2)
            eval(ids)
            #expect(ids.shape == [16, 2])
            let flat = ids.asArray(Int32.self)
            // As strings: Swift tuples are not Equatable, and a hand-rolled zip comparison
            // would be the sort of test-side logic that can disagree with itself.
            let pairs = stride(from: 0, to: flat.count, by: 2).map { "\(flat[$0]),\(flat[$0 + 1])" }

            // First block is the top-left 2x2 — (0,0) (0,1) (1,0) (1,1) — NOT the first row.
            #expect(
                Array(pairs.prefix(4)) == ["0,0", "0,1", "1,0", "1,1"],
                "the first four positions are \(Array(pairs.prefix(4))), which is raster order")
            // Second block is the NEXT 2x2 across, so its columns are 2 and 3.
            #expect(Array(pairs[4 ..< 8]) == ["0,2", "0,3", "1,2", "1,3"])
            // Every coordinate appears exactly once.
            #expect(Set(pairs).count == 16)
        }
    }

    /// A video repeats the same spatial coordinates once per temporal patch.
    @Test("a multi-frame grid repeats the spatial coordinates per frame")
    func videoRepeatsPositions() throws {
        try MLXMetalTestLock.withLock {
            let one = Glm5NextVisionRotary.positionIds(grid: THW(1, 4, 4), mergeSize: 2)
            let three = Glm5NextVisionRotary.positionIds(grid: THW(3, 4, 4), mergeSize: 2)
            eval(one, three)
            #expect(three.shape == [48, 2])
            let a = one.asArray(Int32.self)
            let b = three.asArray(Int32.self)
            #expect(a.count == 32, "one frame of a 4x4 grid is 16 (h, w) pairs")
            #expect(Array(b.prefix(64)) == a + a, "frames must share the spatial layout")
            #expect(Array(b.suffix(32)) == a, "the last frame repeats it too")
        }
    }

    /// THE ONE THAT WOULD HAVE CAUGHT IT.
    ///
    /// Feed the tower a patch sequence, then feed it the same patches with two blocks SWAPPED. A
    /// tower that knows where things are produces something that is not merely the same features in
    /// a different order. A position-blind one produces exactly that, and this fails.
    @Test("shuffling patches changes what they become, not just their order")
    func towerIsPositionSensitive() throws {
        let config = try JSONDecoder().decode(
            Glm5NextConfiguration.self, from: Data(Glm5NextConstructionTests.tinyJSON.utf8))
        try MLXMetalTestLock.withLock {
            let model = try Glm5Next(config, requesting: [.vision])
            let tower = try #require(model.visionTower)

            let grid = THW(1, 4, 4)
            let rowWidth = 3 * 2 * 14 * 14
            let patches = MLXRandom.normal([grid.product, rowWidth])
                .asType(tower.patchEmbed.proj.weight.dtype)

            // Swap the first two merge blocks — four rows each, contiguous because the layout is
            // block-major. Their CONTENT is unchanged; only where they sit is.
            let order = Array(4 ..< 8) + Array(0 ..< 4) + Array(8 ..< grid.product)
            let swapped = take(patches, MLXArray(order.map { Int32($0) }), axis: 0)

            let a = try tower(patches, grid: grid)
            let b = try tower(swapped, grid: grid)
            eval(a, b)
            #expect(a.shape == b.shape)

            // After the merge each block is one output row, so a position-blind tower would give
            // exactly `a` with its first two rows swapped.
            let merged = a.dim(0)
            var expectedIfBlind = Array(0 ..< merged)
            expectedIfBlind.swapAt(0, 1)
            let blind = take(a, MLXArray(expectedIfBlind.map { Int32($0) }), axis: 0)
            eval(blind)

            let difference = abs(b.asType(.float32) - blind.asType(.float32)).max().item(Float.self)
            #expect(
                difference > 1e-3,
                "swapping two blocks only permuted the output (diff \(difference)): position-blind")
        }
    }

}
