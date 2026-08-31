// Copyright © 2026 osaurus-eval contributors
// SPDX-License-Identifier: MIT
//
// The clamped SwiGLU, and the merger's activation — three bugs that produced a model which saw
// the wrong picture while every shape, every parameter and every finiteness check passed.
//
// The reference is three lines:
//
//     gate = gate.clamp(min=None, max=limit)      # RAW gate, UPPER bound only
//     up   = up.clamp(min=-limit, max=limit)      # RAW up, BOTH bounds
//     return down_proj(act_fn(gate) * up)         # activation AFTER the clamp
//
// This file had `clip(silu(gate), -limit, limit) * up`: the wrong quantity, at the wrong point,
// with the wrong bounds, and `up` never clamped at all. And the vision merger dropped its GELU
// entirely, handing the language model a linear projection where a non-linear one was trained.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing

@Suite("GLM-5.3 clamped SwiGLU and merger activation")
struct Glm5NextActivationTests {

    private func value(_ x: MLXArray) -> Float {
        eval(x)
        return x.asType(.float32).asArray(Float.self)[0]
    }

    /// The gate clamp is UPPER-bound only, on the raw projection.
    ///
    /// Each case below is one the previous implementation got wrong, and each is checked against
    /// arithmetic done here from the reference's three lines — not against the implementation.
    @Test("the gate is clamped above only, before the activation")
    func gateClampIsUpperOnly() throws {
        try MLXMetalTestLock.withLock {
            let limit: Float = 10

            // Above the limit: the RAW gate is capped, so the result is silu(10) * up.
            let high = Glm5NextActivation.clampedSwiGLU(
                gate: MLXArray([Float(25)]), up: MLXArray([Float(1)]), limit: limit)
            let siluOfLimit = limit / (1 + exp(-limit))   // 9.99954..., NOT 10
            // 1e-4, deliberately. The old implementation clamped the POST-activation value and so
            // returned exactly `limit`; the gap from silu(limit) is 4.6e-4, which a 1e-3 tolerance
            // would have waved through — and did, until a mutation run showed this case passing
            // with the bug in place.
            #expect(abs(value(high) - siluOfLimit) < 1e-4)

            // FAR BELOW: `min=None` means the negative tail is untouched. Clamping symmetrically —
            // which the old code effectively did on the post-activation value — would change this.
            let low = Glm5NextActivation.clampedSwiGLU(
                gate: MLXArray([Float(-25)]), up: MLXArray([Float(1)]), limit: limit)
            let siluOfMinus25 = Float(-25) / (1 + exp(25))
            #expect(abs(value(low) - siluOfMinus25) < 1e-4)
        }
    }

    /// `up` is clamped SYMMETRICALLY. The old code never clamped it at all.
    @Test("up is clamped on both sides")
    func upIsClampedBothSides() throws {
        try MLXMetalTestLock.withLock {
            let limit: Float = 10
            let siluOfOne = Float(1) / (1 + exp(-1))

            let bigPositive = Glm5NextActivation.clampedSwiGLU(
                gate: MLXArray([Float(1)]), up: MLXArray([Float(400)]), limit: limit)
            #expect(abs(value(bigPositive) - siluOfOne * limit) < 1e-3)

            let bigNegative = Glm5NextActivation.clampedSwiGLU(
                gate: MLXArray([Float(1)]), up: MLXArray([Float(-400)]), limit: limit)
            #expect(abs(value(bigNegative) - siluOfOne * -limit) < 1e-3)
        }
    }

    /// With no limit the clamps must not fire at all.
    @Test("a nil limit is plain SwiGLU")
    func nilLimitIsPlainSwiGLU() throws {
        try MLXMetalTestLock.withLock {
            let out = Glm5NextActivation.clampedSwiGLU(
                gate: MLXArray([Float(25)]), up: MLXArray([Float(400)]), limit: nil)
            let expected = Float(25) / (1 + exp(-25)) * 400
            #expect(abs(value(out) / expected - 1) < 1e-3)
        }
    }

    /// The merger applies GELU between its norm and its SwiGLU.
    ///
    /// Asserted as a DIFFERENCE against the same computation without the activation, built from the
    /// merger's own submodules — so this pins the activation's presence without re-implementing the
    /// module and grading it against itself.
    @Test("the merger's projection is not linear — the GELU is applied")
    func mergerAppliesGelu() throws {
        let config = try JSONDecoder().decode(
            Glm5NextConfiguration.self, from: Data(Glm5NextConstructionTests.tinyJSON.utf8))
        let vision = try #require(config.visionConfig)
        try MLXMetalTestLock.withLock {
            let merger = Glm5NextPatchMerger(vision)
            let x = MLXRandom.normal([4, vision.outHiddenSize])

            let actual = merger(x)
            let normed = merger.postProjectionNorm(merger.proj(x))
            let withoutGelu = merger.down(
                Glm5NextActivation.clampedSwiGLU(
                    gate: merger.gate(normed), up: merger.up(normed),
                    limit: vision.swigluLimit))
            eval(actual, withoutGelu)

            let difference = abs(actual.asType(.float32) - withoutGelu.asType(.float32))
                .max().item(Float.self)
            #expect(
                difference > 1e-4,
                "the merger matches the no-activation path (\(difference)): its GELU is missing")
        }
    }
}
